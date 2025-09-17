import re


def attach_segmentation_wrapper_to_sc(sc, overlap_tokens: int | None = None, overlap_ratio: float = 0.05):  # noqa: PLR0915
    """
    Attach a chunking wrapper to a SelectiveContext instance `sc`.
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
        overlap_tokens = min(max(computed, 5), 128, max_len - 1)

    def safe_get_self_information(text: str) -> tuple[list[str], list[float]]:
        """
        Chunk by sentences first; if a single sentence is still larger than max_len,
        split by token-id windows with overlap. Returns concatenated tokens & infos.
        """
        try:
            enc_whole = sc.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            whole_len = enc_whole["input_ids"].size(1)
        except Exception:
            whole_len = max_len + 1

        if whole_len <= max_len:
            print("Skipped chunking")
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
                win_start = max(0, start - overlap_tokens)
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
