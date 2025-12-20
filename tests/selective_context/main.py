import re
import time

import modal
import tiktoken


def prepare_tokenizer_for_counting_modal(model: str = "gpt-4o"):
    try:
        encoder = tiktoken.encoding_for_model(model)
        print("finished encoder")
        print(f"Successfully loaded encoding {encoder.name}")
    except KeyError:
        print(f"Unknown model '{model}', falling back to cl100k_base.")
        encoder = tiktoken.get_encoding("cl100k_base")

    return encoder


app = modal.App("selective_context")
# image_s = modal.Image.debian_slim(python_version="3.12").pip_install("selective_context", "spacy==3.7.0")
image_s = modal.Image.debian_slim(python_version="3.12").pip_install("selective_context", "spacy==3.8.7", "tiktoken")

image_s = image_s.run_commands("python -m spacy download en_core_web_sm")
with image_s.imports():
    import tiktoken
    from selective_context import SelectiveContext


def attach_safe_wrapper_to_sc(sc, overlap_tokens: int = None, overlap_ratio: float = 0.05):
    """
    Attach a safe chunking wrapper to a SelectiveContext instance `sc`.
    After calling this, sc.get_self_information(text) will work for arbitrarily long text.

    overlap_tokens: explicit integer tokens to overlap between windows.
                    if None, computed as int(overlap_ratio * max_len) and clamped.
    overlap_ratio: fraction of sc.max_token_length to use for default overlap (if overlap_tokens is None).
    """
    base = (
        getattr(sc, "_get_self_info_via_gpt2", None)
        or getattr(sc, "_get_self_info_via_curie", None)
        or sc.get_self_information
    )

    max_len = max(8, sc.max_token_length - 1)

    if overlap_tokens is None:
        computed = int(max(1, round(overlap_ratio * max_len)))
        overlap_tokens = min(max(computed, 5), min(128, max_len - 1))

    def safe_get_self_information(text: str) -> tuple[list[str], list[float]]:
        """
        Chunk by sentences first; if a single sentence is still larger than max_len,
        split by token-id windows with overlap. Returns concatenated tokens & infos.
        """
        try:
            enc_whole = sc.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            whole_len = enc_whole["input_ids"].size(1)
        except Exception:
            # if tokenizer fails to return tensors (rare), force chunking path
            whole_len = max_len + 1

        if whole_len <= max_len:
            return base(text)

        sents = [s.strip() for s in re.split(sc.sent_tokenize_pattern, text) if s.strip()]

        tokens_all: list[str] = []
        infos_all: list[float] = []

        chunk_text = ""
        for sent in sents:
            candidate = chunk_text + (" " if chunk_text else "") + sent
            enc = sc.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")
            if enc["input_ids"].size(1) <= max_len:
                chunk_text = candidate
                continue

            if chunk_text:
                tks, infs = base(chunk_text)
                tokens_all.extend(tks)
                infos_all.extend(infs)

            enc_sent = sc.tokenizer(sent, add_special_tokens=False)
            ids = enc_sent["input_ids"]
            step = max_len - overlap_tokens if overlap_tokens < max_len else max_len
            first_window = True
            for start in range(0, len(ids), step):
                win_start = max(0, start - overlap_tokens)  # include overlap before current window
                window_ids = ids[win_start : win_start + max_len]
                sub_text = sc.tokenizer.decode(window_ids, clean_up_tokenization_spaces=False)
                tks, infs = base(sub_text)

                if not tokens_all:
                    tokens_all.extend(tks)
                    infos_all.extend(infs)
                else:
                    overlap_trim = min(len(tks), overlap_tokens) if not first_window else min(len(tks), overlap_tokens)
                    if overlap_trim > 0:
                        tks = tks[overlap_trim:]
                        infs = infs[overlap_trim:]
                    tokens_all.extend(tks)
                    infos_all.extend(infs)

                first_window = False

            chunk_text = ""  # reset chunk

        if chunk_text:
            tks, infs = base(chunk_text)
            tokens_all.extend(tks)
            infos_all.extend(infs)

        return tokens_all, infos_all

    sc._safe_wrapper = safe_get_self_information
    sc._safe_wrapper_overlap = overlap_tokens
    sc._base_get_self_information = base

    sc.get_self_information = lambda text: sc._safe_wrapper(text)

    return sc


# specify your desired GPU name
@app.function(gpu="A100-80GB", image=image_s)
def main(input_text: str, reduce_ratio: float = 0.5, overlap_tokens=None):
    """
    Compresses `input_text` using Selective Context API while measuring elapsed time.
    Outputs:
      - number of words before/after,
      - compression rate,
      - time taken (sec),
      - compressed context,
      - dropped content (optional).
    """

    print("📦 Initializing SelectiveContext(model_type='gpt2', lang='en')")
    sc = SelectiveContext(model_type="gpt2", lang="en")

    attach_safe_wrapper_to_sc(sc, overlap_tokens=overlap_tokens)

    tokenizer = prepare_tokenizer_for_counting_modal()
    t0 = time.perf_counter()
    reduced_ctx, masked = sc(input_text, reduce_ratio=reduce_ratio)
    t1 = time.perf_counter()

    prompt_tokens = len(tokenizer.encode(input_text))
    compressed_tokens = len(tokenizer.encode(reduced_ctx))
    print(f"Compression finished in {t1 - t0:.2f}s")
    print("TOTAL prompt LEN: ", prompt_tokens)
    print("TOTAL COMPRESSED LEN: ", compressed_tokens)
    print("Reduced context (first 800 chars):\n", reduced_ctx[:800])
    print("\nNumber of masked/filtered items:", len(masked))
    # For debugging you may want to inspect token counts via get_self_information
    try:
        tokens, infos = sc.get_self_information(input_text if len(input_text.split()) < 2000 else input_text[:1000])
        print("Sample token count:", len(tokens))
    except Exception as e:
        print("Sample get_self_information failed:", e)

    print("ORIGINAL", input_text)
    print("Compressed_text", reduced_ctx)
    print("IDK", masked)
    return reduced_ctx, masked


@app.local_entrypoint()
def main_run():
    txt = "Mosquito fish found on the islands of the Bahamas live in various isolated freshwater ponds that were once a single body of water. When several male and female mosquito fish are taken from two isolated ponds and placed into a single pond, the breeding preference of each mosquito fish is for fish from its own original pond. Which of these most likely resulted in this breeding preference? Options: A. Availability of food influenced the breeding preferences of the fish., B. Competition for a suitable mate influenced the breeding preferences., C. Predators in the pond influenced the breeding preferences of the fish., D. Speciation due to reproductive isolation influenced the breeding preferences."
    txt = """
    Exceptional examples of the bourgeois architecture of the later periods were not restored by the communist authorities after the war (like mentioned Kronenberg Palace and Insurance Company Rosja building) or they were rebuilt in socialist realism style (like Warsaw Philharmony edifice originally inspired by Palais Garnier in Paris). Despite that the Warsaw University of Technology building (1899–1902) is the most interesting of the late 19th-century architecture. Some 19th-century buildings in the Praga district (the Vistula’s right bank) have been restored although many have been poorly maintained. Warsaw’s municipal government authorities have decided to rebuild the Saxon Palace and the Brühl Palace, the most distinctive buildings in prewar Warsaw. What style was the Warsaw Philharmony edifice built in?
    """
    txt = """You are given a report by a government agency. Write a one-page summary of the report.

	Report:
	Introduction

	The President is responsible for appointing individuals to certain positions in the federal government. In some instances, the President makes these appointments using authorities granted to the President alone. Other appointments, generally referred to with the abbreviation PAS, are made by the President with the advice and consent of the Senate via the nomination and confirmation process. This report identifies, for the 115 th Congress, all nominations submitted to the Senate for full-time positions on 34 regulatory and other collegial boards and commissions.
	This report includes profiles on the leadership structures of each of these 34 boards and commissions as well as a pair of tables presenting information on each body's membership and appointment activity as of the end of the 115 th Congress.
	The profiles discuss the statutory requirements for the appointed positions, including the number of members on each board or commission, their terms of office, whether they may continue in their positions after their terms expire, whether political balance is required, and the method for selecting the chair. The first table in each pair provides information on full-time positions requiring Senate confirmation as of the end of the 115 th Congress. The second table tracks appointment activity for each board or commission within the 115 th Congress by the Senate (confirmations, rejections, returns to the President, and elapsed time between nomination and confirmation), as well as further related presidential activity (including withdrawals and recess appointments).
	In some instances, no appointment action occurred within a board or commission during the 115 th Congress.
	Information for this report was compiled using the Senate nominations database at https://www.congress.gov/ (users can click the "nominations" tab on the left-hand side of the page to search the database), the Congressional Record (daily edition), the Weekly Compilation of Presidential Documents , telephone discussions with agency officials, agency websites, the United States Code , and the 2016 Plum Book ( United States Government Policy and Supporting Positions ).
	Congressional Research Service (CRS) reports regarding the presidential appointments process, nomination activity for other executive branch positions, recess appointments, and other related matters are available to congressional clients at http://www.crs.gov .

			Characteristics of Regulatory and Other Collegial Bodies

				Common Features

	Federal executive branch boards and commissions discussed in this report share, among other characteristics, the following: (1) they are independent executive branch bodies located, with four exceptions, outside executive departments; (2) several board or commission members head each entity, and at least one of these members serves full time; (3) the members are appointed by the President with the advice and consent of the Senate; and (4) the members serve fixed terms of office and, except in a few bodies, the President's power to remove them is restricted.

				Terms of Office

	For most of the boards and commissions included in this report, the fixed terms of office for member positions have set beginning and end dates, irrespective of whether the posts are filled or when appointments are made. In contrast, for a few agencies, such as the Chemical Safety and Hazard Investigation Board, the full term begins when an appointee takes office and expires after the incumbent has held the post for the requisite period of time. The end dates of the fixed terms of a board's members are staggered, so that the terms do not expire all at once. The use of terms with fixed beginning and end dates is intended to minimize the occurrence of simultaneous board member departures and thereby increase leadership continuity.
	Under such an arrangement, an individual is nominated to a particular position and a particular term of office. An individual may be nominated and confirmed for a position for the remainder of an unexpired term to replace an appointee who has resigned (or died). Alternatively, an individual might be nominated for an upcoming term with the expectation that the new term will be under way by the time of confirmation. Occasionally, when the unexpired term has been for a relatively short period, the President has submitted two nominations of the same person simultaneouslyâthe first to complete the unexpired term and the second to complete the entire succeeding term of office.

				Appointment of Chairs and Political Independence

	On some commissions, the chair is subject to Senate confirmation and must be appointed from among the incumbent commissioners. If the President wishes to appoint, as chair, someone who is not on the commission, the President simultaneously submits two nominations for the nomineeâone for member and the other for chair.
	As independent entities with staggered membership, executive branch boards and commissions have more political independence from the President than do executive departments. Nonetheless, the President can sometimes exercise significant influence over the composition of a board or commission's membership when he designates the chair or has the opportunity to fill a number of vacancies at once. For example, President George W. Bush had the chance to shape the Securities and Exchange Commission (SEC) during the first two years of his presidency because of existing vacancies, resignations, and a member's death. Likewise, during the same time period, President Bush was able to submit nominations for all of the positions on the National Labor Relations Board because of existing vacancies, expiring recess appointments, and resignations. Simultaneous turnover of board or commission membership may result from coincidence, but it also may be the result of a buildup of vacancies after extended periods of time in which the President does not nominate, or the Senate does not confirm, members.

				Political Affiliations and Inspectors General

	Two other notable characteristics apply to appointments to some of the boards and commissions. First, for 26 of the 34 bodies discussed in this report, the law limits the number of appointed members who may belong to the same political party, usually to no more than a bare majority of the appointed members (e.g., two of three or three of five). Second, advice and consent requirements also apply to inspector general appointments in four of these organizations and general counsel appointments in three.

			Appointments During the 115th Congress

	During the 115 th Congress, President Donald Trump submitted nominations to the Senate for 112 of the 151 full-time positions on 34 regulatory and other boards and commissions. In attempting to fill these 112 positions, he submitted a total of 140 nominations, of which 75 were confirmed, 12 were withdrawn, and 53 were returned to the President. No recess appointments were made. Table 1 summarizes the appointment activity for the 115 th Congress. At the end of the Congress, 22 incumbents were serving past the expiration of their terms. In addition, there were 43 vacancies among the 151 positions.

			Length of Time to Confirm a Nomination

	The length of time a given nomination may be pending in the Senate has varied widely. Some nominations have been confirmed within a few days, others have been confirmed within several months, and some have never been confirmed. In the board and commission profiles, this report provides, for each board or commission nomination confirmed in the 115 th Congress, the number of days between nomination and confirmation ("days to confirm").
	Under Senate rules, nominations not acted on by the Senate at the end of a session of Congress (or before a recess of 30 days) are returned to the President. The Senate, by unanimous consent, often waives this ruleâalthough not always. In cases where the President resubmits a returned nomination, this report measures the days to confirm from the date of receipt of the resubmitted nomination, not the original.
	For those nominations confirmed in the 115 th Congress, a mean of 121.0 days elapsed between nomination and confirmation. The median number of days elapsed was 91.0.

		Organization of the Report

			Board and Commission Profiles

	Each of the 34 board or commission profiles in this report is organized into three parts. First, the leadership structure section discusses the statutory requirements for the appointed positions, including the number of members on each board or commission, their terms of office, whether these members may continue in their positions after their terms expire, whether political balance is required, and the method for selecting the chair.
	The first table lists incumbents to full-time positions as of the end of the 115 th Congress, along with party affiliation (where applicable), date of first confirmation, and term expiration date. Incumbents whose terms have expired are italicized. Most incumbents serve fixed terms of office and are removable only for specified causes. They generally remain in office when a new Administration assumes office following a presidential election.
	The second table lists appointment action for vacant positions during the 115 th Congress. This table provides the name of the nominee, position title, date of nomination or appointment, date of confirmation, and number of days between receipt of a nomination and confirmation, and notes relevant actions other than confirmation (e.g., nominations returned to or withdrawn by the President).
	When more than one nominee has had appointment action, the second table also provides statistics on the length of time between nomination and confirmation. The average days to confirm are provided in the form of a mean number.

			Additional Appointment Information

	Appendix A provides two tables. Table A-1 includes information on each of the nominations and appointments to regulatory and other collegial boards and commissions during the 115 th Congress. It is alphabetically organized and follows a similar format to that of the "Appointment Action" sections discussed above. It identifies the board or commission involved and the dates of nomination and confirmation. It also indicates if a nomination was withdrawn, returned, rejected, or if a recess appointment was made. In addition, it provides the mean and median number of days taken to confirm a nomination.
	Table A-2 contains summary information on appointments and nominations by organization. For each of the 34 independent boards and commissions discussed in this report, it shows the number of positions, vacancies, incumbents whose term had expired, nominations, individual nominees, positions to which nominations were made, confirmations, nominations returned to the President, nominations withdrawn, and recess appointments.
	A list of organization abbreviations can be found in Appendix B .

		Chemical Safety and Hazard Investigation Board10

	The Chemical Safety and Hazard Investigation Board is an independent agency consisting of five members who serve five-year terms (no political balance is required), including a chair. The President appoints the members, including the chair, with the advice and consent of the Senate. When a term expires, the incumbent must leave office.

		Commodity Futures Trading Commission11

	The Commodity Futures Trading Commission consists of five members (no more than three may be from the same political party) who serve five-year terms. At the end of a term, a member may remain in office, unless replaced, until the end of the next session of Congress. The chair is also appointed by the President, with the advice and consent of the Senate.

		Consumer Product Safety Commission12

	The statute establishing the Consumer Product Safety Commission calls for five members who serve seven-year terms. No more than three members may be from the same political party. A member may remain in office for one year at the end of a term, unless replaced. The chair is also appointed by the President, with the advice and consent of the Senate.

		Defense Nuclear Facilities Safety Board13

	The Defense Nuclear Facilities Safety Board consists of five members (no more than three may be from the same political party) who serve five-year terms. After a term expires, a member may continue to serve until a successor takes office. The President designates the chair and vice chair.

		Election Assistance Commission14

	The Election Assistance Commission consists of four members (no more than two may be from the same political party) who serve four-year terms. After a term expires, a member may continue to serve until a successor takes office. The chair and vice chair, from different political parties and designated by the commission, change each year.

		Equal Employment Opportunity Commission15

	The Equal Employment Opportunity Commission consists of five members (no more than three may be from the same political party) who serve five-year terms. An incumbent whose term has expired may continue to serve until a successor is appointed, except that no such member may continue to serve (1) for more than 60 days when Congress is in session, unless a successor has been nominated; or (2) after the adjournment of the session of the Senate in which the successor's nomination was submitted. The President designates the chair and the vice chair. The President also appoints the general counsel, with the advice and consent of the Senate.

		Export-Import Bank Board of Directors16

	The Export-Import Bank Board of Directors comprises the bank president, who serves as chair; the bank first vice president, who serves as vice chair; and three other members (no more than three of these five may be from the same political party). All five members are appointed by the President, with the advice and consent of the Senate, and serve for terms of up to four years. An incumbent whose term has expired may continue to serve until a successor is qualified, or until six months after the term expiresâwhichever occurs earlier. The President also appoints an inspector general, with the advice and consent of the Senate.

		Farm Credit Administration18

	The Farm Credit Administration consists of three members (no more than two may be from the same political party) who serve six-year terms. A member may not succeed himself or herself unless he or she was first appointed to complete an unexpired term of three years or less. A member whose term expires may continue to serve until a successor takes office. One member is designated by the President to serve as chair for the duration of the member's term.

		Federal Communications Commission19

	The Federal Communications Commission consists of five members (no more than three may be from the same political party) who serve five-year terms. When a term expires, a member may continue to serve until the end of the next session of Congress, unless a successor is appointed before that time. The President designates the chair.

		Federal Deposit Insurance Corporation BoardÂ ofÂ Directors20

	The Federal Deposit Insurance Corporation Board of Directors consists of five members, of whom twoâthe comptroller of the currency and the director of the Consumer Financial Protection Bureauâare ex officio. The three appointed members serve six-year terms. An appointed member may continue to serve after the expiration of a term until a successor is appointed. Not more than three members of the board may be from the same political party. The President appoints the chair and the vice chair, with the advice and consent of the Senate, from among the appointed members. The chair is appointed for a term of five years. The President also appoints the inspector general, with the advice and consent of the Senate.

		Federal Election Commission22

	The Federal Election Commission consists of six members (no more than three may be from the same political party) who may serve for a single term of six years. When a term expires, a member may continue to serve until a successor takes office. The chair and vice chair, from different political parties and elected by the commission, change each year. Generally, the vice chair succeeds the chair.

		Federal Energy Regulatory Commission23

	The Federal Energy Regulatory Commission, an independent agency within the Department of Energy, consists of five members (no more than three may be from the same political party) who serve five-year terms. When a term expires, a member may continue to serve until a successor takes office, except that such commissioner may not serve beyond the end of the session of the Congress in which his or her term expires. The President designates the chair.

		Federal Labor Relations Authority24

	The Federal Labor Relations Authority consists of three members (no more than two may be from the same political party) who serve five-year terms. After the date on which a five-year term expires, a member may continue to serve until the end of the next Congress, unless a successor is appointed before that time. The President designates the chair. The President also appoints the general counsel, with the advice and consent of the Senate.

		Federal Maritime Commission25

	The Federal Maritime Commission consists of five members (no more than three may be from the same political party) who serve five-year terms. When a term expires, a member may continue to serve until a successor takes office. The President designates the chair.

		Federal Mine Safety and Health Review Commission26

	The Federal Mine Safety and Health Review Commission consists of five members (no political balance is required) who serve six-year terms. When a term expires, the member must leave office. The President designates the chair.

		Federal Reserve System Board of Governors27

	The Federal Reserve System Board of Governors consists of seven members (no political balance is required) who serve 14-year terms. When a term expires, a member may continue to serve until a successor takes office. The President appoints the chair and vice chair, who are separately appointed as members, for four-year terms, with the advice and consent of the Senate.

		Federal Trade Commission28

	The Federal Trade Commission consists of five members (no more than three may be from the same political party) who serve seven-year terms. When a term expires, the member may continue to serve until a successor takes office. The President designates the chair.

		Financial Stability Oversight Council29

	The Financial Stability Oversight Council consists of 10 voting members and 5 nonvoting members, and is chaired by the Secretary of the Treasury. Of the 10 voting members, 9 serve ex officio, by virtue of their positions as leaders of other agencies. The remaining voting member is appointed by the President with the advice and consent of the Senate and serves full time for a term of six years. Of the five nonvoting members, two serve ex officio. The remaining three nonvoting members are designated through a process determined by the constituencies they represent, and they serve for terms of two years. The council is not required to have a balance of political party representation.

		Foreign Claims Settlement Commission30

	The Foreign Claims Settlement Commission, located in the Department of Justice, consists of three members (political balance is not required) who serve three-year terms. When a term expires, the member may continue to serve until a successor takes office. Only the chair, who is appointed by the President with the advice and consent of the Senate, serves full time.

		Merit Systems Protection Board31

	The Merit Systems Protection Board consists of three members (no more than two may be from the same political party) who serve seven-year terms. A member who has been appointed to a full seven-year term may not be reappointed to any following term. When a term expires, the member may continue to serve for one year, unless a successor is appointed before that time. The President appoints the chair, with the advice and consent of the Senate, and designates the vice chair.

		National Credit Union Administration BoardÂ ofÂ Directors32

	The National Credit Union Administration Board of Directors consists of three members (no more than two members may be from the same political party) who serve six-year terms. When a term expires, a member may continue to serve until a successor takes office. The President designates the chair.

		National Labor Relations Board33

	The National Labor Relations Board consists of five members who serve five-year terms. Political balance is not required, but, by tradition, no more than three members are from the same political party. When a term expires, the member must leave office. The President designates the chair. The President also appoints the general counsel, with the advice and consent of the Senate.

		National Mediation Board34

	The National Mediation Board consists of three members (no more than two may be from the same political party) who serve three-year terms. When a term expires, the member may continue to serve until a successor takes office. The board annually designates a chair.

		National Transportation Safety Board35

	The National Transportation Safety Board consists of five members (no more than three may be from the same political party) who serve five-year terms. When a term expires, a member may continue to serve until a successor takes office. The President appoints the chair from among the members for a two-year term, with the advice and consent of the Senate, and designates the vice chair.

		Nuclear Regulatory Commission36

	The Nuclear Regulatory Commission consists of five members (no more than three may be from the same political party) who serve five-year terms. When a term expires, the member must leave office. The President designates the chair. The President also appoints the inspector general, with the advice and consent of the Senate.

		Occupational Safety and Health ReviewÂ Commission38

	The Occupational Safety and Health Review Commission consists of three members (political balance is not required) who serve six-year terms. When a term expires, the member must leave office. The President designates the chair.

		Postal Regulatory Commission39

	The Postal Regulatory Commission consists of five members (no more than three may be from the same political party) who serve six-year terms. After a term expires, a member may continue to serve until his or her successor takes office, but the member may not continue to serve for more than one year after the date upon which his or her term otherwise would expire. The President designates the chair, and the members select the vice chair.

		Privacy and Civil Liberties Oversight Board40

	The Privacy and Civil Liberties Oversight Board consists of five members (no more than three may be from the same political party) who serve six-year terms. When a term expires, the member may continue to serve until a successor takes office. Only the chair, who is appointed by the President with the advice and consent of the Senate, serves full time.
	The Implementing Recommendations of the 9/11 Commission Act of 2007, P.L. 110-53 , Title VIII, Section 801 (121 Stat. 352), established the Privacy and Civil Liberties Oversight Board. Previously, the Privacy and Civil Liberties Oversight Board functioned as part of the White House Office in the Executive Office of the President. That board ceased functioning on January 30, 2008.

		Railroad Retirement Board41

	The Railroad Retirement Board consists of three members (political balance is not required) who serve five-year terms. When a term expires, the member may continue to serve until a successor takes office. The President appoints the chair and an inspector general with the advice and consent of the Senate.

		Securities and Exchange Commission43

	The Securities and Exchange Commission consists of five members (no more than three may be from the same political party) who serve five-year terms. When a term expires, the member may continue to serve until the end of the next session of Congress, unless a successor is appointed before that time. The President designates the chair.

		Surface Transportation Board44

	The Surface Transportation Board, located within the Department of Transportation, consists of five members (no more than three may be from the same political party) who serve five-year terms. When a term expires, the member may continue to serve until a successor takes office but for not more than one year after expiration. The President designates the chair.

		United States International Trade Commission45

	The United States International Trade Commission consists of six members (no more than three may be from the same political party) who serve nine-year terms. A member of the commission who has served for more than five years is ineligible for reappointment. When a term expires, a member may continue to serve until a successor takes office. The President designates the chair and vice chair for two-year terms of office, but they may not belong to the same political party. The President may not designate a chair with less than one year of continuous service as a member. This restriction does not apply to the vice chair.

		United States Parole Commission

	The United States Parole Commission is an independent agency in the Department of Justice. The commission consists of five commissioners (political balance is not required) who serve for six-year terms. When a term expires, a member may continue to serve until a successor takes office. In most cases, a commissioner may serve no more than 12 years. The President designates the chair (18 U.S.C. Â§4202). The commission was previously scheduled to be phased out, but Congress has extended its life several times. Under P.L. 113-47 , Section 2 (127 Stat. 572), it was extended until November 1, 2018 (18 U.S.C. Â§3551 note).

		United States Sentencing Commission46

	The United States Sentencing Commission is a judicial branch agency that consists of seven voting members, who are appointed to six-year terms, and one nonvoting member. The seven voting members are appointed by the President, with the advice and consent of the Senate, and only the chair and three vice chairs serve full time. The President appoints the chair, with the advice and consent of the Senate, and designates the vice chairs. At least three members must be federal judges. No more than four members may be of the same political party. No more than two vice chairs may be of the same political party. No voting member may serve more than two full terms. When a term expires, an incumbent may continue to serve until he or she is reappointed, a successor takes office, or Congress adjourns sine die at the end of the session that commences after the expiration of the term, whichever is earliest. The Attorney General (or designee) serves ex officio as a nonvoting member. The chair of the United State Parole Commission also is an ex officio nonvoting member of the commission.
	Appendix A. Summary of All Nominations and Appointments to Collegial Boards andÂ Commissions
	Appendix B. Board and Commission Abbreviations

	SUMMARY:"""

    txt = """Despite their prevalence, Euclidean embeddings of data are fundamentally limited in their ability to capture latent semantic structures, which need not conform to Euclidean spatial assumptions. Here we consider an alternative, which embeds data as discrete probability distributions in a Wasserstein space, endowed with an optimal transport metric. Wasserstein spaces are much larger and more flexible than Euclidean spaces, in that they can successfully embed a wider variety of metric structures. We propose to exploit this flexibility by learning an embedding that captures the semantic information in the Wasserstein distance between embedded distributions. We examine empirically the representational capacity of such learned Wasserstein embeddings, showing that they can embed a wide variety of complex metric structures with smaller distortion than an equivalent Euclidean embedding. We also investigate an application to word embedding, demonstrating a unique advantage of Wasserstein embeddings: we can directly visualize the high-dimensional embedding, as it is a probability distribution on a low-dimensional space. This obviates the need for dimensionality reduction techniques such as t-SNE for visualization. Learned embeddings form the basis for many state-of-the-art learning systems. Word embeddings like word2vec BID34 , GloVe BID42 , fastText BID5 , and ELMo BID43 are ubiquitous in natural language processing, where they are used for tasks like machine translation BID38 , while graph embeddings BID41 like node2vec BID21 are used to represent knowledge graphs and pre-trained image models BID47 appear in many computer vision pipelines.An effective embedding should capture the semantic structure of the data with high fidelity, in a way that is amenable to downstream tasks. This makes the choice of a target space for the embedding important, since different spaces can represent different types of semantic structure. The most common choice is to embed data into Euclidean space, where distances and angles between vectors encode their levels of association BID34 BID56 BID27 BID36 . Euclidean spaces, however, are limited in their ability to represent complex relationships between inputs, since they make restrictive assumptions about neighborhood sizes and connectivity. This drawback has been documented recently for tree-structured data, for example, where spaces of negative curvature are required due to exponential scaling of neighborhood sizes BID39 BID49 .In this paper, we embed input data as probability distributions in a Wasserstein space. Wasserstein spaces endow probability distributions with an optimal transport metric, which measures the distance traveled in transporting the mass in one distribution to match another. Recent theory has shown that Wasserstein spaces are quite flexible-more so than Euclidean spaces-allowing a variety of other metric spaces to be embedded within them while preserving their original distance metrics. As such, they make attractive targets for embeddings in machine learning, where this flexibility might capture complex relationships between objects when other embeddings fail to do so.Unlike prior work on Wasserstein embeddings, which has focused on embedding into Gaussian distributions BID37 BID58 , we embed input data as discrete distributions supported at a fixed number of points. In doing so, we attempt to access the full flexibility of Wasserstein spaces to represent a wide variety of structures.Optimal transport metrics and their gradients are costly to compute, requiring the solution of a linear program. For efficiency , we use an approximation to the Wasserstein distance called the Sinkhorn divergence BID15 , in which the underlying transport problem is regularized to make it more tractable. While less well-characterized theoretically with respect to embedding capacity, the Sinkhorn divergence is computed efficiently by a fixed-point iteration. Moreover, recent work has shown that it is suitable for gradient-based optimization via automatic differentiation BID20 . To our knowledge, our work is the first to explore embedding properties of the Sinkhorn divergence.We empirically investigate two settings for Wasserstein embeddings. First, we demonstrate their representational capacity by embedding a variety of complex networks, for which Wasserstein embeddings achieve higher fidelity than both Euclidean and hyperbolic embeddings. Second, we compute Wasserstein word embeddings , which show retrieval performance comparable to existing methods. One major benefit of our embedding is that the distributions can be visualized directly, unlike most embeddings, which require a dimensionality reduction step such as t-SNE before visualization. We demonstrate the power of this approach by visualizing the learned word embeddings. Several characteristics determine the value and effectiveness of an embedding space for representation learning. The space must be large enough to embed a variety of metrics, while admitting a mathematical description compatible with learning algorithms; additional features, including direct interpretability, make it easier to understand, analyze, and potentially debug the output of a representation learning procedure. Based on their theoretical properties, Wasserstein spaces are strong candidates for representing complex semantic structures, when the capacity of Euclidean space does not suffice. Empirically, entropy-regularized Wasserstein distances are effective for embedding a wide variety of semantic structures, while enabling direct visualization of the embedding.Our work suggests several directions for additional research. Beyond simple extensions like weighting points in the point cloud, one observation is that we can lift nearly any representation space X to distributions over that space W(X ) represented as point clouds; in this paper we focused on the case X = R n . Since X embeds within W(X ) using δ-functions, this might be viewed as a general "lifting" procedure increasing the capacity of a representation. We can also consider other tasks, such as co-embedding of different modalities into the same transport space. Additionally, our empirical results suggest that theoretical study of the embedding capacity of Sinkhorn divergences may be profitable. Finally, following recent work on computing geodesics in Wasserstein space BID45 , it may be interesting to invert the learned mappings and use them for interpolation."""
    txt = """Recently several different deep learning architectures have been proposed that take a string of characters as the raw input signal and automatically derive features for text classification. Little studies are available that compare the effectiveness of these approaches for character based text classification with each other. In this paper we perform such an empirical comparison for the important cybersecurity problem of DGA detection: classifying domain names as either benign vs. produced by malware (i.e., by a Domain Generation Algorithm). Training and evaluating on a dataset with 2M domain names shows that there is surprisingly little difference between various convolutional neural network (CNN) and recurrent neural network (RNN) based architectures in terms of accuracy, prompting a preference for the simpler architectures, since they are faster to train and less prone to overfitting. Malware is software that infects computers in order to perform unauthorized malicious activities. In order to successfully achieve its goals, the malware needs to be able to connect to a command and control (C&C) center. To this end, both the controller behind the C&C center (hereafter called botmaster) and the malware on the infected machines can run a Domain Generation Algorithm (DGA) that generates hundreds or even thousands of domains automatically. The malware then attempts at resolving each one of these domains with its local DNS server. The botmaster will have registered one or a few of these automatically generated domains. For these domains that have been actually registered, the malware will obtain a valid IP address and will be able to communicate with the C&C center.The binary text classification task that we address in this paper is: given a domain name string as input, classify it as either malicious, i.e. generated by a DGA, or as benign. Deep neural networks have recently appeared in the literature on DGA detection ; BID8 ; BID15 . They significantly outperform traditional machine learning methods in accuracy, at the price of increasing the complexity of training the model and requiring larger datasets. Independent of the work on deep networks for DGA detection, other deep learning approaches for character based text classification have recently been proposed, including deep neural network architectures designed for processing and classification of tweets BID2 ; BID11 ) as well as general natural language text BID16 ). No systematic study is available that compares the predictive accuracy of all these different character based deep learning architectures, leaving one to wonder which one works best for DGA detection.To answer this open question, in this paper we compare the performance of five different deep learning architectures for character based text classification (see TAB0 ) for the problem of detecting DGAs. They all rely on character-level embeddings, and they all use a deep learning architecture based on convolutional neural network (CNN) layers, recurrent neural network (RNN) layers, or a combination of both. Our most important finding is that for DGA detection, which can be thought of as classification of short character strings, despite of vast differences in the deep network architectures, there is remarkably little difference among the methods in terms of accuracy and false positive rates, while they all comfortably outperform a random forest trained on human engineered features. This finding is of practical value for the design of deep neural network based classifiers for short text classification in industry and academia: it provides evidence that one can select an architecture that BID16 is faster to train, without loss of accuracy. In the context of DGA detection, optimizing the training time is of particular importance, as the models need to be retrained on a regular basis to stay current with respect to new, emerging malware. DGA detection, i.e. the classification task of distinguishing between benign domain names and those generated by malware (Domain Generation Algorithms), has become a central topic in information security. In this paper we have compared five different deep neural network architectures that perform this classification task based purely on the domain name string, given as a raw input signal at character level. All five models, i.e. two RNN based architectures, two CNN based architectures, and one hybrid RNN/CNN architecture perform equally well, catching around 97-98% of malicious domain names against a false positive rate of 0.001. This roughly means that for every 970 malicious domain names that the deep networks catch, they flag only one benign domain name erroneously as malicious. A Random Forest based on human defined linguistic features achieves a recall of only 83% against the same 0.001 false positive rate when trained and tested on the same data that was used for the deep networks. The use of a deep neural network that automatically learns features is attractive in a cybersecurity setting because it is a lot harder to craft malware to avoid detection by a system that relies on automatically learned features instead of on human engineered features. An interesting direction for future work is to test the trained deep networks more extensively on domain names generated by new and previously unseen malware families.A KERAS CODE FOR DEEP NETWORKS main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) lstm = LSTM(128, return sequences=False)(embedding) drop = Dropout(0.5) (lstm) output = Dense(1, activation ='sigmoid') (drop) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam')Listing 1: Endgame model with single LSTM layer, adapted from main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) bi lstm = Bidirectional ( layer =LSTM(64, return sequences=False), merge mode='concat')(embedding) output = Dense(1, activation ='sigmoid') ( bi lstm ) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 2: CMU model with bidirectional LSTM, adapted from BID2 main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) conv1 = Conv1D( filters =128, kernel size =3, padding='same', strides =1)(embedding) thresh1 = ThresholdedReLU(1e−6)(conv1) max pool1 = MaxPooling1D(pool size=2, padding='same')(thresh1 ) conv2 = Conv1D( filters =128, kernel size =2, padding='same', strides =1)(max pool1) thresh2 = ThresholdedReLU(1e−6)(conv2) max pool2 = MaxPooling1D(pool size=2, padding='same')(thresh2 ) flatten = Flatten () (max pool2) fc = Dense(64)( flatten ) thresh fc = ThresholdedReLU(1e−6)(fc) drop = Dropout(0.5) ( thresh fc ) output = Dense(1, activation ='sigmoid') (drop) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 3: NYU model with stacked CNN layers, adapted from BID16 def getconvmodel( self , kernel size , filters ) : model = Sequential () model.add( Conv1D( filters = filters , input shape =(128, 128), kernel size = kernel size , padding='same', activation =' relu ', strides =1)) model.add(Lambda(lambda x: K.sum(x, axis=1), output shape =( filters , ) ) ) model.add(Dropout(0.5) ) return model main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) conv1 = getconvmodel(2, 256)(embedding) conv2 = getconvmodel(3, 256)(embedding) conv3 = getconvmodel(4, 256)(embedding) conv4 = getconvmodel(5, 256)(embedding) merged = Concatenate () ([ conv1, conv2, conv3, conv4] ) middle = Dense(1024, activation =' relu ') (merged) middle = Dropout(0.5) (middle) middle = Dense(1024, activation =' relu ') (middle) middle = Dropout(0.5) (middle) output = Dense(1, activation ='sigmoid') (middle) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 4: Invincea CNN model with parallel CNN layers, adapted from BID8 main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) conv = Conv1D( filters =128, kernel size =3, padding='same', activation =' relu ', strides =1)(embedding) max pool = MaxPooling1D(pool size=2, padding='same')(conv) encode = LSTM(64, return sequences=False) (max pool) output = Dense(1, activation ='sigmoid') (encode) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 5: MIT model with a stacked CNN and LSTM layer, adapted from BID11 main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) flatten = Flatten () (embedding) output = Dense(1, activation ='sigmoid') ( flatten ) model = Model(inputs=main input, outputs =output) print (model.summary()) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 6: Baseline Model with only Embedding Layer main input = Input (shape=(11, ) , name='main input') dense = Dense(128, activation =' relu ') ( main input ) output = Dense(1, activation ='sigmoid') (dense) model = Model(inputs=main input, outputs =output) print (model.summary()) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 7: MLP Model with 128 Nodes Dense Layer"""
    main.remote(input_text=txt, reduce_ratio=0.5)
