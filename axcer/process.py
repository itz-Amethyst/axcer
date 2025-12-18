import json
import os
import string
import threading
from collections import Counter, defaultdict
from functools import singledispatchmethod
from itertools import chain, product
from pathlib import Path
from typing import Any, Literal

import nltk
import polars as pl

from axcer.constants.question_words import QUESTION_WORDS
from axcer.graph import GraphProcessor
from axcer.tokenizer import RegexTokenizer
from axcer.utils import count_tokens, get_next_id, load_existing_data, logger, track_runtime_and_memory
from experiments.constants.paths import PROCESSED_AXCER_PATH, fill_path

Word = str
Sentence = str
Phrase = list[str]


class Axcer:
    """Resource-Agnostic text prompt compression."""

    def __init__(
        self,
        stopwords: set[str] | None = None,
        punctuations: set[str] | None = None,
        language: str = "english",
        min_length: int = 1,
        metric_file_name: str = "test",
        interrogative_window_size: int = 4,
        enable_metrics_logging: bool = True,
        exceptionals: list[str] | None = None,
        metrics_file_path: Path | None = "",
        # include_repeated_phrases: bool = True,
    ):
        """Initialize the axcer object with custom options."""

        if punctuations is None:
            punctuations = set("—")

        self.signs = [
            "?",
            "!",
            "%",
            ":",
            ".",
            ";",
            "$",
            "£",  # British Pound
            "$",  # US Dollar
            "€",  # Euro
            "¥",  # Japanese Yen
            "₹",  # Indian Rupee
            "$",  # Canadian Dollar
            "₣",  # French Franc (historically used before Euro)
            "₱",  # Philippine Peso
            "₺",  # Turkish Lira
            "₩",  # South Korean Won
            "₴",  # Ukrainian Hryvnia
            "₼",  # Azerbaijani Manat
            "₭",  # Laotian Kip
            "₮",  # Mongolian Tugrik
            "₨",  # Sri Lankan Rupee, Nepalese Rupee
            "₽",  # Russian Ruble
        ]

        self.excluded_punctuations = {"+", "-", "/", "^", "*", "$", "!", "%", "#"}

        if stopwords:
            self.stopwords = stopwords
        else:
            print("Downloading NLTK stopwords")
            self.stopwords = set(nltk.corpus.stopwords.words(language))
            print("Finished")

        # self.punctuations = (punctuations or set(string.punctuation)) - self.excluded_punctuations
        self.punctuations = (set(string.punctuation) | (punctuations or set())) - self.excluded_punctuations
        self.exceptionals = set(exceptionals) if exceptionals else set()

        if self.exceptionals:
            self.to_ignore = self.stopwords.union(self.punctuations) - self.exceptionals
        else:
            self.to_ignore = self.stopwords.union(self.punctuations)
        self.exceptionals = list(self.exceptionals)

        self.word_tokenizer = RegexTokenizer()

        self.min_length = min_length
        self.max_length = interrogative_window_size
        self.graph_processor = GraphProcessor()

        # NOT a useful feature
        # self.include_repeated_phrases = include_repeated_phrases

        self.frequency_dist: dict[Word, int] = {}
        self.degree: dict[Word, int] = {}
        self.rank_list: list[tuple[float, Sentence]] = []
        self.combined_segments: list[Sentence] = []
        self.results: list[Sentence] = []
        self.enable_metrics_logging = enable_metrics_logging
        if self.enable_metrics_logging:
            self.total_runtime = None
            self.runtime_date = None
            self.total_memory_kb = None
            self.cpu_percent = None
            self.metrics_file_path = (
                metrics_file_path if metrics_file_path else fill_path(PROCESSED_AXCER_PATH, dataset_name=metric_file_name)
            )

    def merge_path_dictionaries(
        self, original_dict: dict[int, list[list[str], list[str]]], extracted_paths
    ) -> dict[int, list[str]]:
        """
        Merge extracted paths into the original dictionary

        Args:
            original_dict (dict): The original dictionary to be updated
            extracted_paths (dict): dictionary of extracted paths from cycles

        Returns:
            dict: Updated dictionary with merged and prioritized paths
        """
        merged_dict = original_dict.copy()

        for key, extracted_words in extracted_paths.items():
            merged_dict[key] = extracted_words + merged_dict.get(key, [])

        # TODO: TYPE FIX
        return merged_dict

    def reset(self):
        self.frequency_dist: dict[Word, int] = {}
        self.degree: dict[Word, int] = {}
        self.rank_list: list[tuple[float, Sentence]] = []
        self.combined_segments: list[Sentence] = []

    def extract_phrases_from_dictionary(self, filtered_phrases_with_indexes: dict[int, list[str]]) -> list[str]:
        """
        Extract phrases from a dictionary while preserving the order of keys.

        Args:
            filtered_phrases_with_indexes (dict): dictionary with index keys and list of phrases

        Returns:
            list: Flattened list of phrases in the order of dictionary keys
        """
        phrases = []

        for key in sorted(filtered_phrases_with_indexes.keys()):
            phrases.extend(filtered_phrases_with_indexes[key])

        return phrases

    def is_two_word_phrase(self, phrase: Phrase) -> bool:
        """Check if the given phrase consists of exactly two words."""
        return len(phrase) == 2

    def _is_valid_phrase(self, phrase: Phrase) -> bool:
        """Check if a phrase is within the valid length range and contains no stopwords."""
        # return self.min_length <= len(phrase) <= self.max_length and all(word not in self.to_ignore for word in phrase)
        return self.min_length <= len(phrase) <= self.max_length and all(word.lower() not in self.to_ignore for word in phrase)

    def _slice_with_indices(self, sample, indices, step):
        result = {}
        index_list = list(range(indices[0], indices[1], step))

        for i, start in enumerate(index_list):
            if i + 1 < len(index_list):
                result[start] = sample[start : index_list[i + 1]]
            else:
                result[start] = sample[start : indices[-1]]
        return result

    def save_processing_metrics(self, original_text: str, save_method: Literal["json", "csv"] = "csv") -> None:
        ext = ".json" if save_method == "json" else ".csv"
        self.metrics_file_path = self.metrics_file_path.with_suffix(ext)

        self.metrics_file_path.parent.mkdir(exist_ok=True, parents=True)
        input_tokens = count_tokens(original_text)

        detokenized_str = self.word_tokenizer.detokenize(self.combined_segments[0])
        # print("Original", original_text)
        # print('DETOKENIZED', detokenized_str)
        compressed_tokens: int = count_tokens(detokenized_str)
        token_saved: int = input_tokens - compressed_tokens
        n_detected_cycle = (
            self.graph_processor.number_of_detected_cycle if self.graph_processor.number_of_detected_cycle >= 1 else 1
        )
        compression_ratio = input_tokens / compressed_tokens
        gpt_o1_saving = (input_tokens - compressed_tokens) * 0.015 / 1000
        avg_distance = self.graph_processor.total_distance_to_interrogative_word / n_detected_cycle
        new_entry: dict[str, str | int | float | None] = {
            "original_text": original_text,
            "compressed_text": detokenized_str,
            "prompt_tokens": input_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": token_saved,
            "compression_time": self.total_runtime,
            "compression_ratio": float(f"{compression_ratio:.3f}"),
            "runtime_date": self.runtime_date,
            "gpt_o1_saving": float(f"{gpt_o1_saving:.2f}"),
            "memory_used_kb": self.total_memory_kb,
            "avg_distance_interrogative_words": float(f"{avg_distance:.3f}") if n_detected_cycle >= 1 else 0.0,
        }

        if save_method == "json":
            existing_data: dict[str, Any] = load_existing_data(self.metrics_file_path)
            next_id: str = get_next_id(existing_data)

            existing_data[next_id] = new_entry

            with open(self.metrics_file_path, "w") as f:
                json.dump(existing_data, fp=f, indent=2)
        elif save_method == "csv":
            new_df = pl.DataFrame(new_entry)

            if os.path.exists(self.metrics_file_path):
                existing_df = pl.read_csv(self.metrics_file_path)
                updated_df = existing_df.vstack(new_df)
                updated_df.write_csv(self.metrics_file_path)
            else:
                new_df.write_csv(self.metrics_file_path)

        logger.info(f"Metrics saved to {self.metrics_file_path}")

    def clear_json_file(self, json_path: Path | None = None) -> None:
        if json_path is None:
            json_path = self.json_path
        json_path.write_text("{}")

    def _generate_phrases(self, sentence: Sentence) -> dict[int, list[str]]:
        """Generate candidate phrases from the text by splitting sentences.
        Return phrases along with their index.
        """
        phrases = {}

        words = self.word_tokenizer.tokenize(sentence)
        # print("Tokenized", words)
        self.tokenized_words = words
        phrase = []

        for index, word in enumerate(words):
            if len(word) >= 1 and word.lower() not in self.signs:
                if word.lower() in self.to_ignore:
                    if phrase:
                        if self._is_valid_phrase(phrase):
                            id = index - len(phrase)
                            phrases[id] = phrase
                        phrase = []
                elif word.lower() in self.exceptionals:
                    logger.info(f"Hit exceptional word : {word}")
                    phrase.append(word)

                elif len(phrase) < self.max_length:
                    phrase.append(word)
                else:
                    id = index - len(phrase)
                    phrases[id] = phrase
                    phrase = []
                    phrase.append(word)

            else:
                word_before_punctuation = words[index - 1]

                if word_before_punctuation.isalnum():
                    if word_before_punctuation not in self.signs:
                        if phrase and phrase[-1].lower() == word_before_punctuation.lower():
                            index = index - len(phrase)
                            phrase.append(word)
                            phrases[index] = phrase
                            phrase = []
                        else:
                            phrase = [word_before_punctuation, word]
                            index = index - 1
                            phrases[index] = phrase
                            phrase = []

        if phrase and self._is_valid_phrase(phrase):
            last_index = len(words) - len(phrase)
            phrases[last_index] = phrase

        return phrases

    # Optional: methods
    def build_frequency_dist(self, phrases: list[Phrase]) -> None:
        """Build frequency distribution of words in phrases."""
        self.frequency_dist = Counter(chain(*phrases))

    def build_word_co_occurrence_graph(self, phrases: list[Phrase]) -> None:
        """Build the word co-occurrence graph to calculate word degrees."""
        word_co_occurrences: defaultdict[Word, set[Word]] = defaultdict(set)
        for phrase in phrases:
            for w1, w2 in product(phrase, repeat=2):
                if w1 != w2:
                    word_co_occurrences[w1].add(w2)

        self.degree = {
            word: len(co_occurrences) + self.frequency_dist[word] for word, co_occurrences in word_co_occurrences.items()
        }

    def _combine_segments(self, phrases_with_indexes: dict[int, Phrase]):
        """
        Rejoin segments to a combined string based on their order in phrases_with_indexes.

        Args:
            phrases_with_indexes (dict[int, List[Phrase]]): Phrases organized by indexes
        """

        def merge_close_tokens(tokens):
            """Merge tokens that should be together."""
            if len(tokens) <= 1:
                return " ".join(map(str, tokens))

            result = []
            current_group = [tokens[0]]

            for token in tokens[1:]:
                token = str(token)
                if len(token) <= 2 or token in string.punctuation:
                    if token not in "-":
                        current_group.append(token)
                else:
                    result.append(" ".join(map(str, current_group)))
                    current_group = [token]

            result.append("".join(map(str, current_group)))
            return " ".join(result)

        temp_list: list[Sentence] = []
        for index in sorted(phrases_with_indexes.keys()):
            tokens = phrases_with_indexes[index]
            merged_phrase = merge_close_tokens(tokens)

            if merged_phrase.strip():
                temp_list.append(merged_phrase)

        self.combined_segments.append(temp_list)

    @track_runtime_and_memory
    def compress_prompt(self, text: str):
        """Main method to compress given prompt."""
        phrases_with_indexes = self._generate_phrases(text)
        self.graph_processor.reset()

        filtered_phrases_with_indexes = []
        temp_dict = {}
        checked_indices = set()
        last_index = -self.max_length - 1
        end_of_last_index = None

        for index, phrase in phrases_with_indexes.items():
            target_is_in_question = threading.Event()
            root = phrase[0]
            if root.lower() not in QUESTION_WORDS:
                start = max(0, end_of_last_index + 1) if end_of_last_index else max(0, index - self.max_length)

                if start == last_index:
                    start += 1

                preceding_segments = {}
                preceding_segment = self.tokenized_words[start:index]
                remaining = index - start
                if remaining > self.max_length:
                    preceding_segments = self._slice_with_indices(self.tokenized_words, [start, index], self.max_length)
                else:
                    preceding_segments[index] = preceding_segment
                    temp_dict.update(preceding_segments)

            else:
                target_is_in_question.set()
                start = index + len(phrase)

            if index in checked_indices:
                last_index = index
                continue

            if start <= index:
                for num in range(start, index + 1):
                    checked_indices.add(num)

            elif target_is_in_question.is_set():
                checked_indices.add(index)
                logger.info("Skipped new upfront")
                last_index = index
                end_of_last_index = (index - 1) + len(phrase)
                continue

            for root_id, segment in preceding_segments.items():
                self.graph_processor.build_cycles(segment, root, root_id)

            last_index = index
            end_of_last_index = (index - 1) + len(phrase)

        interrogative_dictionary = self.graph_processor.extract_interrogative_phrases()

        filtered_phrases_with_indexes = self.merge_path_dictionaries(phrases_with_indexes, interrogative_dictionary)

        self._combine_segments(filtered_phrases_with_indexes)

        if self.enable_metrics_logging:
            self._set_runtime_now()
            self._set_memory_now()
            self.save_processing_metrics(text)

        return self.get_ranked_phrases()

    @singledispatchmethod
    def process_compress_prompt(self, texts) -> str | list:
        raise TypeError(f"Unsupported type: {type(texts)}")

    @process_compress_prompt.register(str)
    def _(self, text: str) -> str:
        return self.compress_prompt(text)

    @process_compress_prompt.register(list)
    def _(self, texts: list | tuple) -> list[str]:
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All list items must be strings")
        for t in texts:
            self.compress_prompt(t)

        return self.print_results()

    def get_combined_segments(self) -> list[Sentence]:
        """Return ranked keyword phrases."""
        results = []
        for item in self.combined_segments:
            detokenized = self.word_tokenizer.detokenize(item)
            results.append(detokenized)
        self.results.extend(results)
        self.reset()
        return results

    def print_results(self) -> list[str]:
        return self.results


if __name__ == "__main__":
    # dataset_name = "test"
    # result_path = fill_path(PROCESSED_AXCER_PATH, dataset_name=dataset_name)
    # r = Axcer(metrics_file_path=result_path)
    r = Axcer()

    text42 = """(CNN) -- Dennis Farina, the dapper, mustachioed cop-turned-actor best known for his tough-as-nails work in such TV series as "Law & Order," "Crime Story," and "Miami Vice," has died. He was 69. "We are deeply saddened by the loss of a great actor and a wonderful man," said his publicist, Lori De Waal, in a statement Monday. "Dennis Farina was always warmhearted and professional, with a great sense of humor and passion for his profession. He will be greatly missed by his family, friends and colleagues." Farina, who had a long career as a police officer in Chicago, got into acting through director Michael Mann, who used him as a consultant and cast him in his 1981 movie, "Thief." That role led to others in such Mann-created shows as "Miami Vice" (in which Farina played a mobster) and "Crime Story" (in which he starred as Lt. Mike Torello). Farina also had roles, generally as either cops or gangsters, in a number of movies, including "Midnight Run" (1988), "Get Shorty" (1995), "The Mod Squad" (1999) and "Snatch" (2000). In 2004, he joined the cast of the long-running "Law & Order" after Jerry Orbach's departure, playing Detective Joe Fontana, a role he reprised on the spinoff "Trial by Jury." Fontana was known for flashy clothes and an expensive car, a distinct counterpoint to Orbach's rumpled Lennie Briscoe. Farina was on "Law & Order" for two years, partnered with Jesse L. Martin's Ed Green. Martin's character became a senior detective after Farina left the show. """
    text42 = """Backdoor attacks aim to manipulate a subset of training data by injecting adversarial triggers such that machine learning models trained on the tampered dataset will make arbitrarily (targeted) incorrect prediction on the testset with the same trigger embedded.", "While federated learning (FL) is capable of aggregating information provided by different parties for training a better model, its distributed learning methodology and inherently heterogeneous data distribution across parties may bring new vulnerabilities.", "In addition to recent centralized backdoor attacks on FL where each party embeds the same global trigger during training, we propose the distributed backdoor attack (DBA) --- a novel threat assessment framework developed by fully exploiting the distributed nature of FL.", "DBA decomposes a global trigger pattern into separate local patterns and embed them into the training set of different adversarial parties respectively.", "Compared to standard centralized backdoors, we show that DBA is substantially more persistent and stealthy against FL on diverse datasets such as finance and image data.", "We conduct extensive experiments to show that the attack success rate of DBA is significantly higher than centralized backdoors under different settings.", "Moreover, we find that distributed attacks are indeed more insidious, as DBA can evade two state-of-the-art robust FL algorithms against centralized backdoors.", "We also provide explanations for the effectiveness of DBA via feature visual interpretation and feature importance ranking.\n", "To further explore the properties of DBA, we test the attack performance by varying different trigger factors, including local trigger variations (size, gap, and location), scaling factor in FL, data distribution, and poison ratio and interval.", "Our proposed DBA and thorough evaluation results shed lights on characterizing the robustness of FL.", "Federated learning (FL) has been recently proposed to address the problems for training machine learning models without direct access to diverse training data, especially for privacy-sensitive tasks (Smith et al., 2017; McMahan et al., 2017; Zhao et al., 2018) .", "Utilizing local training data of participants (i.e., parties), FL helps train a shared global model with improved performance.", "There have been prominent applications and ever-growing trends in deploying FL in practice, such as loan status prediction, health situation assessment (e.g. potential cancer risk assessment), and next-word prediction while typing (Hard et al., 2018; Yang et al., 2018; 2019) .", "Although FL is capable of aggregating dispersed (and often restricted) information provided by different parties to train a better model, its distributed learning methodology as well as inherently heterogeneous (i.e., non-i.i.d.) data distribution across different parties may unintentionally provide a venue to new attacks.", "In particular, the fact of limiting access to individual party's data due to privacy concerns or regulation constraints may facilitate backdoor attacks on the shared model trained with FL.", "Backdoor attack is a type of data poisoning attacks that aim to manipulate a subset of training data such that machine learning models trained on the tampered dataset will be vulnerable to the test set with similar trigger embedded (Gu et al., 2019) .", "Backdoor attacks on FL have been recently studied in (Bagdasaryan et al., 2018; Bhagoji et al., 2019) .", "However, current attacks do not fully exploit the distributed learning methodology of FL, as they embed the same global trigger pattern to all adversarial parties.", "We call such attacking scheme Figure 1: Overview of centralized and distributed backdoor attacks (DBA) on FL.", "The aggregator at round t + 1 combines information from local parties (benign and adversarial) in the previous round t, and update the shared model G t+1 .", "When implementing backdoor attacks, centralized attacker uses a global trigger while distributed attacker uses a local trigger which is part of the global one.", "centralized backdoor attack.", "Leveraging the power of FL in aggregating dispersed information from local parties to train a shared model, in this paper we propose distributed backdoor attack (DBA) against FL.", "Given the same global trigger pattern as the centralized attack, DBA decomposes it into local patterns and embed them to different adversarial parties respectively.", "A schematic comparison between the centralized and distributed backdoor attacks is illustrated in Fig.1 .", "Through extensive experiments on several financial and image datasets and in-depth analysis, we summarize our main contributions and findings as follows.", "• We propose a novel distributed backdoor attack strategy DBA on FL and show that DBA is more persistent and effective than centralized backdoor attack.", "Based on extensive experiments, we report a prominent phenomenon that although each adversarial party is only implanted with a local trigger pattern via DBA, their assembled pattern (i.e., global trigger) attains significantly better attack performance on the global model compared with the centralized attack.", "The results are consistent across datasets and under different attacking scenarios such as one-time (single-shot) and continuous (multiple-shot) poisoning settings.", "To the best of our knowledge, this paper is the first work studying distributed backdoor attacks.", "• When evaluating the robustness of two recent robust FL methods against centralized backdoor attack (Fung et al., 2018; Pillutla et al., 2019) , we find that DBA is more effective and stealthy, as its local trigger pattern is more insidious and hence easier to bypass the robust aggregation rules.", "• We provide in-depth explanations for the effectiveness of DBA from different perspectives, including feature visual interpretation and feature importance ranking.", "• We perform comprehensive analysis and ablation studies on several trigger factors in DBA, including the size, gap, and location of local triggers, scaling effect in FL, poisoning interval, data poisoning ratio, and data distribution.", "Specifically, at round t, the central server sends the current shared model G t to n ∈ [N ] selected parties, where [N ] denotes the integer set {1, 2, . . . , N }.", "The selected party i locally computes the function f i by running an optimization algorithm such as stochastic gradient descent (SGD) for E local epochs with its own dataset D i and learning rate l r to obtain a new local model L t+1 i", ". The local party then sends model update L t+1 i − G t back to the central server, who will averages over all updates with its own learning rate η to generate a new global model G t+1 :", "This aggregation process will be iterated until FL finds the final global model.", "Unless specified otherwise, we use G t (L t i ) to denote the model parameters of the global (local) model at round t.", "Attacker ability.", "Based on the Kerckhoffs's theory (Shannon, 1949) , we consider the strong attacker here who has full control of their local training process, such as backdoor data injection and updating local training hyperparameters including E and l r .", "This scenario is quite practical since each local dataset is usually owned by one of the local parties.", "However, attackers do not have the ability to influence the privilege of central server such as changing aggregation rules, nor tampering the training process and model updates of other parties.", "Objective of backdoor attack.", "Backdoor attack is designed to mislead the trained model to predict a target label τ on any input data that has an attacker-chosen pattern (i.e., a trigger) embedded.", "Instead of preventing the convergence in accuracy as Byzantine attacks (Blanchard et al., 2017) , the purpose of backdoor attacks in FL is to manipulate local models and simultaneously fit the main task and backdoor task, so that the global model would behave normally on untampered data samples while achieving high attack success rate on backdoored data samples.", "The adversarial objective for attacker i in round t with local datatset D i and target label τ is:", "Here, the poisoned dataset", "The function R transforms clean data in any class into backdoored data that have an attacker-chosen trigger pattern using a set of parameters φ.", "For example, for image data, φ is factored into trigger location TL, trigger size TS and trigger gap TG (φ = {TS, TG, TL}), which are shown in Fig.2 .", "The attacker can design his own trigger pattern and choose an optimal poison ratio r to result in a better model parameter w * i , with which G t+1 can both assign the highest probability to target label τ for backdoored data R(x i j , φ) and the ground truth label y i j for benign data x i j .", "Through extensive experiments on diverse datasets including LOAN and three image datasets in different settings, we show that in standard FL our proposed DBA is more persistent and effective than centralized backdoor attack: DBA achieves higher attack success rate, faster convergence and better resiliency in single-shot and multiple-shot attack scenarios.", "We also demonstrate that DBA is more stealthy and can successfully evade two robust FL approaches.", "The effectiveness of DBA is explained using feature visual interpretation for inspecting its role in aggregation.", "We also perform an in-depth analysis on the important factors that are unique to DBA to explore its properties and limitations.", "Our results suggest DBA is a new and more powerful attack on FL than current backdoor attacks.", "Our analysis and findings can provide new threat assessment tools and novel insights for evaluating the adversarial robustness of FL.", "A APPENDIX"""
    text42 = "A research team finds a new species of animal. The information learned about the animal would best further scientific knowledge if the research team Options: A. issued a press release to local television stations., B. presented information to the scientific community., C. discussed the scientific findings with students., D. wrote a letter to the editor of its local newspaper."
    text42 = """Although variational autoencoders (VAEs) represent a widely influential deep generative model, many aspects of the underlying energy function remain poorly understood. In particular, it is commonly believed that Gaussian encoder/decoder assumptions reduce the effectiveness of VAEs in generating realistic samples . In this regard, we rigorously analyze the VAE objective, differentiating situations where this belief is and is not actually true . We then leverage the corresponding insights to develop a simple VAE enhancement that requires no additional hyperparameters or sensitive tuning . Quantitatively, this proposal produces crisp samples and stable FID scores that are actually competitive with a variety of GAN models, all while retaining desirable attributes of the original VAE architecture . The code for our model is available at \\url{https://github.com/daib13/TwoStageVAE}. Our starting point is the desire to learn a probabilistic generative model of observable variables x ∈ χ, where χ is a r-dimensional manifold embedded in R d . Note that if r = d, then this assumption places no restriction on the distribution of x ∈ R d whatsoever; however, the added formalism is introduced to handle the frequently encountered case where x possesses low-dimensional structure relative to a high-dimensional ambient space, i.e., r d. In fact, the very utility of generative models of continuous data, and their attendant low-dimensional representations, often hinges on this assumption BID1 . It therefore behooves us to explicitly account for this situation.Beyond this, we assume that χ is a simple Riemannian manifold, which means there exists a diffeomorphism ϕ between χ and R r , or more explicitly, the mapping ϕ : χ → R r is invertible and differentiable. Denote a ground-truth probability measure on χ as µ gt such that the probability mass of an infinitesimal dx on the manifold is µ gt (dx) and χ µ gt (dx) = 1.The variational autoencoder (VAE) BID17 BID28 attempts to approximate this ground-truth measure using a parameterized density p θ (x) defined across all of R d since any underlying generative manifold is unknown in advance. This density is further assumed to admit the latent decomposition p θ (x) = p θ (x|z)p(z)dz, where z ∈ R κ serves as a lowdimensional representation, with κ ≈ r and prior p(z) = N (z|0, I).Ideally we might like to minimize the negative log-likelihood − log p θ (x) averaged across the ground-truth measure µ gt , i.e., solve min θ χ − log p θ (x)µ gt (dx). Unfortunately though, the required marginalization over z is generally infeasible. Instead the VAE model relies on tractable encoder q φ (z|x) and decoder p θ (x|z) distributions, where φ represents additional trainable parameters. The canonical VAE cost is a bound on the average negative log-likelihood given by L(θ, φ) χ {− log p θ (x) + KL [q φ (z|x)||p θ (z|x)]} µ gt (dx) ≥ χ − log p θ (x)µ gt (dx),where the inequality follows directly from the non-negativity of the KL-divergence. Here φ can be viewed as tuning the tightness of bound, while θ dictates the actual estimation of µ gt . Using a few standard manipulations, this bound can also be expressed as DISPLAYFORM0 which explicitly involves the encoder/decoder distributions and is conveniently amenable to SGD optimization of {θ, φ} via a reparameterization trick BID17 BID28 . The first term in (2 ) can be viewed as a reconstruction cost (or a stochastic analog of a traditional autoencoder), while the second penalizes posterior deviations from the prior p(z). Additionally, for any realizable implementation via SGD, the integration over χ must be approximated via a finite sum across training samples {x (i) } n i=1 drawn from µ gt . Nonetheless, examining the true objective L(θ, φ) can lead to important, practically-relevant insights.At least in principle, q φ (z|x) and p θ (x|z) can be arbitrary distributions, in which case we could simply enforce q φ (z|x) = p θ (z|x) ∝ p θ (x|z)p(z) such that the bound from (1) is tight. Unfortunately though, this is essentially always an intractable undertaking. Consequently, largely to facilitate practical implementation, a commonly adopted distributional assumption for continuous data is that both q φ (z|x) and p θ (x|z) are Gaussian. This design choice has previously been cited as a key limitation of VAEs BID5 BID18 , and existing quantitative tests of generative modeling quality thus far dramatically favor contemporary alternatives such as generative adversarial networks (GAN) BID13 . Regardless, because the VAE possesses certain desirable properties relative to GAN models (e.g., stable training BID29 , interpretable encoder/inference network BID4 , outlier-robustness BID9 , etc.), it remains a highly influential paradigm worthy of examination and enhancement.In Section 2 we closely investigate the implications of VAE Gaussian assumptions leading to a number of interesting diagnostic conclusions. In particular, we differentiate the situation where r = d, in which case we prove that recovering the ground-truth distribution is actually possible iff the VAE global optimum is reached, and r < d, in which case the VAE global optimum can be reached by solutions that reflect the ground-truth distribution almost everywhere, but not necessarily uniquely so. In other words, there could exist alternative solutions that both reach the global optimum and yet do not assign the same probability measure as µ gt .Section 3 then further probes this non-uniqueness issue by inspecting necessary conditions of global optima when r < d. This analysis reveals that an optimal VAE parameterization will provide an encoder/decoder pair capable of perfectly reconstructing all x ∈ χ using any z drawn from q φ (z|x). Moreover, we demonstrate that the VAE accomplishes this using a degenerate latent code whereby only r dimensions are effectively active. Collectively, these results indicate that the VAE global optimum can in fact uniquely learn a mapping to the correct ground-truth manifold when r < d, but not necessarily the correct probability measure within this manifold, a critical distinction.Next we leverage these analytical results in Section 4 to motivate an almost trivially-simple, twostage VAE enhancement for addressing typical regimes when r < d. In brief, the first stage just learns the manifold per the allowances from Section 3, and in doing so, provides a mapping to a lower dimensional intermediate representation with no degenerate dimensions that mirrors the r = d regime. The second (much smaller) stage then only needs to learn the correct probability measure on this intermediate representation, which is possible per the analysis from Section 2. Experiments from Sections 5 and 6 empirically corroborate motivational theory and reveal that the proposed two-stage procedure can generate high-quality samples, reducing the blurriness often attributed to VAE models in the past BID11 BID21 . And to the best of our knowledge, this is the first demonstration of a VAE pipeline that can produce stable FID scores, an influential recent metric for evaluating generated sample quality BID16 , that are comparable to GAN models under neutral testing conditions. Moreover, this is accomplished without additional penalties, cost function modifications, or sensitive tuning parameters. Finally, an extended version of this work can be found in BID8 ). There we include additional results, consideration of disentangled representations, as well as a comparative discussion of broader VAE modeling paradigms such as those involving normalizing flows or parameterized families for p(z). It is often assumed that there exists an unavoidable trade-off between the stable training, valuable attendant encoder network, and resistance to mode collapse of VAEs, versus the impressive visual quality of images produced by GANs. While we certainly are not claiming that our two-stage VAE model is superior to the latest and greatest GAN-based architecture in terms of the realism of generated samples, we do strongly believe that this work at least narrows that gap substantially such that VAEs are worth considering in a broader range of applications. For further results and discussion, including consideration of broader VAE modeling paradigms and the identifiability of disentangled representations, please see BID8 ."""
    text42 = """Recently several different deep learning architectures have been proposed that take a string of characters as the raw input signal and automatically derive features for text classification. Little studies are available that compare the effectiveness of these approaches for character based text classification with each other. In this paper we perform such an empirical comparison for the important cybersecurity problem of DGA detection: classifying domain names as either benign vs. produced by malware (i.e., by a Domain Generation Algorithm). Training and evaluating on a dataset with 2M domain names shows that there is surprisingly little difference between various convolutional neural network (CNN) and recurrent neural network (RNN) based architectures in terms of accuracy, prompting a preference for the simpler architectures, since they are faster to train and less prone to overfitting. Malware is software that infects computers in order to perform unauthorized malicious activities. In order to successfully achieve its goals, the malware needs to be able to connect to a command and control (C&C) center. To this end, both the controller behind the C&C center (hereafter called botmaster) and the malware on the infected machines can run a Domain Generation Algorithm (DGA) that generates hundreds or even thousands of domains automatically. The malware then attempts at resolving each one of these domains with its local DNS server. The botmaster will have registered one or a few of these automatically generated domains. For these domains that have been actually registered, the malware will obtain a valid IP address and will be able to communicate with the C&C center.The binary text classification task that we address in this paper is: given a domain name string as input, classify it as either malicious, i.e. generated by a DGA, or as benign. Deep neural networks have recently appeared in the literature on DGA detection ; BID8 ; BID15 . They significantly outperform traditional machine learning methods in accuracy, at the price of increasing the complexity of training the model and requiring larger datasets. Independent of the work on deep networks for DGA detection, other deep learning approaches for character based text classification have recently been proposed, including deep neural network architectures designed for processing and classification of tweets BID2 ; BID11 ) as well as general natural language text BID16 ). No systematic study is available that compares the predictive accuracy of all these different character based deep learning architectures, leaving one to wonder which one works best for DGA detection.To answer this open question, in this paper we compare the performance of five different deep learning architectures for character based text classification (see TAB0 ) for the problem of detecting DGAs. They all rely on character-level embeddings, and they all use a deep learning architecture based on convolutional neural network (CNN) layers, recurrent neural network (RNN) layers, or a combination of both. Our most important finding is that for DGA detection, which can be thought of as classification of short character strings, despite of vast differences in the deep network architectures, there is remarkably little difference among the methods in terms of accuracy and false positive rates, while they all comfortably outperform a random forest trained on human engineered features. This finding is of practical value for the design of deep neural network based classifiers for short text classification in industry and academia: it provides evidence that one can select an architecture that BID16 is faster to train, without loss of accuracy. In the context of DGA detection, optimizing the training time is of particular importance, as the models need to be retrained on a regular basis to stay current with respect to new, emerging malware. DGA detection, i.e. the classification task of distinguishing between benign domain names and those generated by malware (Domain Generation Algorithms), has become a central topic in information security. In this paper we have compared five different deep neural network architectures that perform this classification task based purely on the domain name string, given as a raw input signal at character level. All five models, i.e. two RNN based architectures, two CNN based architectures, and one hybrid RNN/CNN architecture perform equally well, catching around 97-98% of malicious domain names against a false positive rate of 0.001. This roughly means that for every 970 malicious domain names that the deep networks catch, they flag only one benign domain name erroneously as malicious. A Random Forest based on human defined linguistic features achieves a recall of only 83% against the same 0.001 false positive rate when trained and tested on the same data that was used for the deep networks. The use of a deep neural network that automatically learns features is attractive in a cybersecurity setting because it is a lot harder to craft malware to avoid detection by a system that relies on automatically learned features instead of on human engineered features. An interesting direction for future work is to test the trained deep networks more extensively on domain names generated by new and previously unseen malware families.A KERAS CODE FOR DEEP NETWORKS main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) lstm = LSTM(128, return sequences=False)(embedding) drop = Dropout(0.5) (lstm) output = Dense(1, activation ='sigmoid') (drop) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam')Listing 1: Endgame model with single LSTM layer, adapted from main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) bi lstm = Bidirectional ( layer =LSTM(64, return sequences=False), merge mode='concat')(embedding) output = Dense(1, activation ='sigmoid') ( bi lstm ) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 2: CMU model with bidirectional LSTM, adapted from BID2 main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) conv1 = Conv1D( filters =128, kernel size =3, padding='same', strides =1)(embedding) thresh1 = ThresholdedReLU(1e−6)(conv1) max pool1 = MaxPooling1D(pool size=2, padding='same')(thresh1 ) conv2 = Conv1D( filters =128, kernel size =2, padding='same', strides =1)(max pool1) thresh2 = ThresholdedReLU(1e−6)(conv2) max pool2 = MaxPooling1D(pool size=2, padding='same')(thresh2 ) flatten = Flatten () (max pool2) fc = Dense(64)( flatten ) thresh fc = ThresholdedReLU(1e−6)(fc) drop = Dropout(0.5) ( thresh fc ) output = Dense(1, activation ='sigmoid') (drop) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 3: NYU model with stacked CNN layers, adapted from BID16 def getconvmodel( self , kernel size , filters ) : model = Sequential () model.add( Conv1D( filters = filters , input shape =(128, 128), kernel size = kernel size , padding='same', activation =' relu ', strides =1)) model.add(Lambda(lambda x: K.sum(x, axis=1), output shape =( filters , ) ) ) model.add(Dropout(0.5) ) return model main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) conv1 = getconvmodel(2, 256)(embedding) conv2 = getconvmodel(3, 256)(embedding) conv3 = getconvmodel(4, 256)(embedding) conv4 = getconvmodel(5, 256)(embedding) merged = Concatenate () ([ conv1, conv2, conv3, conv4] ) middle = Dense(1024, activation =' relu ') (merged) middle = Dropout(0.5) (middle) middle = Dense(1024, activation =' relu ') (middle) middle = Dropout(0.5) (middle) output = Dense(1, activation ='sigmoid') (middle) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 4: Invincea CNN model with parallel CNN layers, adapted from BID8 main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) conv = Conv1D( filters =128, kernel size =3, padding='same', activation =' relu ', strides =1)(embedding) max pool = MaxPooling1D(pool size=2, padding='same')(conv) encode = LSTM(64, return sequences=False) (max pool) output = Dense(1, activation ='sigmoid') (encode) model = Model(inputs=main input, outputs =output) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 5: MIT model with a stacked CNN and LSTM layer, adapted from BID11 main input = Input (shape=(75, ) , dtype='int32 ', name='main input') embedding = Embedding(input dim=128, output dim=128, input length =75)(main input ) flatten = Flatten () (embedding) output = Dense(1, activation ='sigmoid') ( flatten ) model = Model(inputs=main input, outputs =output) print (model.summary()) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 6: Baseline Model with only Embedding Layer main input = Input (shape=(11, ) , name='main input') dense = Dense(128, activation =' relu ') ( main input ) output = Dense(1, activation ='sigmoid') (dense) model = Model(inputs=main input, outputs =output) print (model.summary()) model.compile( loss =' binary crossentropy ', optimizer ='adam') Listing 7: MLP Model with 128 Nodes Dense Layer"""

    text42 = "Sentence1: The service is deploying Cisco 's BTS 10200 Softswitch cable modem termination system and MGXR 8850 voice gateway products . Sentence2: This solution includes the BTS 10200 soft switch , uBR7246VXR cable modem termination system and MGX 8850 voice gateway products ."
    text42 = """
We introduce a new dataset of logical entailments for the purpose of measuring models' ability to capture and exploit the structure of logical expressions against an entailment prediction task. We use this task to compare a series of architectures which are ubiquitous in the sequence-processing literature, in addition to a new model class---PossibleWorldNets---which computes entailment as a ``convolution over possible worlds''. Results show that convolutional networks present the wrong inductive bias for this class of problems relative to LSTM RNNs, tree-structured neural networks outperform LSTM RNNs due to their enhanced ability to exploit the syntax of logic, and PossibleWorldNets outperform all benchmarks. This paper seeks to answer two questions: "Can neural networks understand logical formulae well enough to detect entailment?", and, more generally, "Which architectures are best at inferring, encoding, and relating features in a purely structural sequence-based problem?". In answering these questions, we aim to better understand the inductive biases of popular architectures with regard to structure and abstraction in sequence data. Such understanding would help pave the road to agents and classifiers that reason structurally, in addition to reasoning on the basis of essentially semantic representations. In this paper, we provide a testbed for evaluating some aspects of neural networks' ability to reason structurally and abstractly. We use it to compare a variety of popular network architectures and a new model we introduce, called PossibleWorldNet.Neural network architectures lie at the heart of a variety of applications. They are practically ubiquitous across vision tasks BID19 BID17 BID29 and natural language understanding, from machine translation BID13 BID33 BID2 to textual entailment BID3 BID27 via sentiment analysis BID31 BID14 and reading comprehension BID9 BID25 . They have been used to synthesise programs BID20 BID23 BID5 or internalise algorithms BID6 BID11 BID12 BID26 . They form the basis of reinforcement learning agents capable of playing video games BID22 , difficult perfect information games BID28 BID35 , and navigating complex environments from raw pixels BID21 ). An important question in this context is to find the inductive and generalisation properties of different neural architectures, particularly towards the ability to capture structure present in the input, an ability that might be important for many language and reasoning tasks. However, there is little work on studying these inductive biases in isolation by running these models on tasks that are primarily or purely about sequence structure, which we intend to address. The paper's contribution is three-fold. First, we introduce a new dataset for training and evaluating models. Second, we provide a thorough evaluation of the existing neural models on this dataset. Third, inspired by the semantic (model-theoretic) definition of entailment, we propose a variant of the TreeNet that evaluates the formulas in multiple different "possible worlds", and which significantly outperforms the benchmarks. The structure of this paper is as follows. In Section 2, we introduce the new dataset and describe a generic data generation process for entailment datasets, which offers certain guarantees against the presence of superficial exploitable biases. In Section 3, we describe a series of baseline models used to validate the dataset, benchmarks from which we will derive our analyses of popular model architectures, and also introduce our new neural model, the PossibleWorldNet. In Section 4, we describe the structure of experiments, from which we obtained the results presented and discussed in Section 5. We offer a brief survey of related work in Section 6, before making concluding remarks in Section 7. Experimental results are shown in TAB1 . The test scores of the best performing overall model are indicated in bold. The test scores of the best performing model which does not have privileged access to the syntax or semantics of the logic (i.e. excluding TreeRNN-based models) are italicised. The best benchmark test results are underlined.We observe that the baselines are doing better than random (8.2 points above for the easy test set, for the MLP BoW, and 2.6 above random for the hard test set). This indicates that there are some small number of exploitable regularities at the symbolic level in this dataset, but that they do not provide significant information.The baseline results show that convolution networks and BiDirLSTMs encoders obtain relatively mediocre results compared to other models, as do LSTM and BiDirLSTM Traversal models. LSTM encoders is the best performing model which does not have privileged access to the syntax trees. Their success relative to BiDirLSTMs Encoders could be due to their reduced number of parameters guarding against overfitting, and rendering them easier to optimise, but it is plausible BiDirLSTMs Encoders would perform similarly with a more fine-grained grid search. Both tree-based models take the lead amongst the benchmarks, with the TreeLSTM being the best performing benchmark overall on both test sets. For most models except baselines, the symbol permutation data augmentation yielded 2-3 point increase in accuracy on weaker models (BiDirLSTM encoders and traversals, an convolutional networks) and between 7-15 point increases for the Tree-based models. This indicates that this data augmentation strategy is particularly well fitted for letting structure-aware models capture, at the representational level, the arbitrariness of symbols indicating unbound variables.Overall, these results show clearly that models that exploit structure in problems where it is provided, unambiguous, and a central feature of the task, outperform models which must implicitly model the structure of sequences. LSTM-based encoders provide robust and competitive results, although bidirectionality is not necessarily always the obvious choice due to optimisation and overfitting problems. Perhaps counter-intuitively, given the results of BID27 , traversal models do not outperform encoding models in this pair-of-sequences traversal problem, indicating that they may be better at capturing the sort of long-range dependencies need to recognise textual entailment better than they are at capturing structure in general.We conclude, from these benchmark results, that tree structured networks may be a better choice for domains with unambiguous syntax, such as analysing formal languages or programs. For domains such as natural language understanding, both convolutional and recurrent network architectures have had some success, but our experiments indicate that this may be due to the fact that existing tasks favour models which capture representational or semantic regularities, and do not adequately test for structural or syntactic reasoning. In particular, the poor performance of convolutional nets on this task serves as a useful indicator that while they present the right inductive bias for capturing structure in images, where topological proximity usually indicates a joint semantic contribution (pixels close by are likely to contribute to the same "part" of an image, such as an edge or pattern), this inductive bias does not carry over to sequences particularly well (where dependencies may be significantly more sparse, structured, and distant) § . The results for the transformer benchmark indicate that while this architecture can capture sufficient structure for machine translation, allowing for the appropriate word order in the output, and accounting for disambiguation or relational information where it exists within sentences, it does not capture with sufficient precision the more hierarchical structure which exists in logical expressions.The best performing model overall is the PossibleWorldNet, which achieves significantly higher results than the other models, with 99.3% accuracy on test (easy), and 97.3% accuracy on test (hard). This is as to be expected, as it has the strongest inductive bias. This inductive bias has two components. First, the model has knowledge of the syntactic structure of the expression, since it is a variant of a TreeNet. Second, inspired by the definition of semantic (model-theoretic) entailment in § Related to this point, BID15 show that convolutional networks make for good character-level encoders, to produce word representations, which are in turn better exploited by RNNs. This is consistent with our interpretation of our results, since at the character level, topological distance is-like for images-a good indicator of semantic grouping (characters that are close are usually part of the same word or n-gram). general, the model evaluates the pair of formulas in lots of different situations ("possible worlds") and combines the various results together in a product ¶ .The quality of the PossibleWorldNet depends directly on the number of "possible worlds" it considers (see FIG1 . As we increase the number of possible worlds, the validation error rate goes down steadily. Note that the data-efficiency also increases as we increase the number of worlds. This is because adding worlds to the model does not increase the number of model parameters-it just increases the number of different "possibilities" that are considered. In propositional logic, of course, if we are allowed to generate every single truth-value assignment, then it is trivial to detect entailment by checking each one. In our big test set, there are on average more than 3,000 possible truth-value assignments. In our massive test set, there are on average over 800,000 possible assignments. (See TAB0 ). The PossibleWorldNet considers at most 256 different worlds, which is only 7% of the expected total number of rows needed in the big test set, and only 0.03% of the expected number of rows needed for the massive test set.To understand this result, we sample 32, 64, 128 and 256 truth table rows (variable truth-value assignments) for each pair of formulas in Test (hard), and reject entailment if a single evaluation for the formulas amongst these finds the left hand side to be true while the right hand side is false. This gives us an estimate of the accuracy of sampling a number of truth table rows equal to the number of possible worlds in our model. We estimate that these statistical methods have 75.9%, 86.5%, 93.4% and 97.2% chance of finding a countermodel, respectively. This seems to indicate that PossibleWorldNet is capable of exploiting repeated computation across projections of random noise in order to learn, solely based on the label likelihood objective, something akin to a modelbased solution to entailment by treating the random-noise as variable valuations.6 RELATED WORK BID39 show how a neural architecture can be used to optimise matrix expressions. They generate all expressions up to a certain depth , group them into equivalence classes, and train a recursive neural network classifier to detect whether two expressions are in the same equivalence class. They use a recursive neural network BID30 to guide the search for an optimised equivalent expression. There are two major differences between this work and ours. First, the classifier is predicting whether two matrix expressions (e.g. A and (A T ) T ) compute the same values; this is an equivalence relation, while entailment is a partial order. Second, their dataset consists of matrix expressions containing at most one variable, while our formulas contain many variables. BID1 use a recursive neural network to learn whether two expressions are equivalent. They tested on two datasets: propositional logic and polynomials. There are two main differences between their approach and ours. First, we consider entailment while they consider equivalence; equivalence is a symmetric relation, while entailment is not symmetric. Second, we consider entailment as a relational classification problem: given a pair of expressions A and B, predict whether A entails B. In their paper, by contrast, they generate a set of k equivalence-classes of ¶ See Formula 2 above. This general notion of entailment as truth-in-all-worlds is not dependent on any particular formal logic, and applies to entailment in both formal logics and natural languages.formulas with the same truth-conditions, and ask the network to predict which of these k classes a single formula falls into. Their task is more specific: their network is only able to classify a formula from a new equivalence class that has not been seen during training if it has additional auxiliary information about that class (e.g. exemplar members of the class).Recognizing textual entailment (RTE) between natural language sentences is a central task in natural language processing. (See Dagan et al. (2006) ; for a recent dataset, see BID3 ). Some approaches (e.g., BID37 and BID27 ) use LSTMs with attention, while others (e.g. , BID38 ) use a convolutional neural network with attention. Of course, recognizing entailment between natural language sentences is a very different task from recognizing entailment between logical formulas. Evaluating an entailment between natural language sentences requires understanding the meaning of the non-logical terms in the sentence. For example, the inference from "An ice skating rink placed outdoors is full of people" to "A lot of people are in an ice skating park" requires knowing the non-logical semantic information that an outdoors ice skating rink is also an ice skating park.Current neural models do not always understand the structure of the sentences they are evaluating. In BID3 , all the neural models they considered wrongly claimed that "A man wearing padded arm protection is being bitten by a German shepherd dog" entails "A man bit a dog". We believe that isolating the purely structural sub-problem will be useful because only networks that can reliably predict entailment in a purely formal setting, such as propositional (or first-order) logic, will be capable of getting these sorts of examples consistently correct. In this paper, we have introduced a new process for generating datasets for the purpose of recognising logical entailment. This was used to compare benchmarks and a new model on a task which is primarily about understanding and exploiting structure. We have established two clear results on the basis of this task. First, and perhaps most intuitively, architectures which make explicit use of structure will perform significantly better than those which must implicitly capture it. Second, the best model is the one that has a strong architectural bias towards capturing the possible world semantics of entailment. In addition to these two points, experimental results also shed some light on the relative abilities of implicit structure models-namely LSTM and Convolution networkbased architectures-to capture structure, showing that convolutional networks may not present the right inductive bias to capture and exploit the heterogeneous and deeply structured syntax in certain sequence-based problems, both for formal and natural languages.This conclusion is to be expected: the most successful models are those with the most prior knowledge about the generic structure of the task at hand. But our dataset throws new light on this unsurprising thought, by providing a new data-point on which to evaluate neural models' ability to understand structural sequence problems. Logical entailment, unlike textual entailment, depends only on the meaning of the logical operators, and of the place particular arbitrarily-named variables hold within a structure. Here, we have a task in which a network's understanding of structure can be disentangled from its understanding of the meaning of words.Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In Advances in neural information processing systems, pp. 649-657, 2015.Xiaodan Zhu, Parinaz Sobihani, and Hongyu Guo. Long short-term memory over recursive structures. In International Conference on Machine Learning, pp. 1604 Learning, pp. -1612 Learning, pp. , 2015 A THE DATASET A.1 DATASET REQUIREMENTS Our dataset D is composed of triples of the form (A, B, A B) , where A B is 1 if A entails B, and 0 otherwise. For
example: (p ∧ q, q, 1) (q ∨ r, r, 0)We wanted to ensure that simple baseline models are unable to exploit simple statistical regularities to perform well in this task. We define a series of baseline models which, due to their structure or the information they have access to, should not be able to solve the entailment recognition problem described in this paper. We distinguish baselines for which we believe there is little chance of them detecting entailment, from those for which there categorically cannot be true modelling of entailment. The baselines which categorically cannot detect entailment are encoding models which only observe one side of the sequent: DISPLAYFORM0 where f is a linear bag of words encoder, an MLP bag of words encoder, or a TreeNet.Because the dataset contains a roughly balanced number of positive and negative examples, it follows that we should expect any model which only sees part of the sequent to perform in line with a random classifier. If they outperform a random baseline on test, there is a structural or symbolic regularity on one side (or both) which is sufficient to identify some subset of positive or negative examples. We use these baselines to verify the soundness of the generation process.Let D + and D − be the positive and negative entailments: DISPLAYFORM1 We impose various requirements on the dataset, to rule out superficial syntactic differences between D + and D − that can be easily exploited by the simple baselines described above. We require that our classes are balanced: and , we are guaranteed to produce balanced classes. Unfortunately, this straightforward approach generates datasets that violate most of our requirements above. See TAB3 for the details. DISPLAYFORM2 In particular, the mean number of negations, conjunctions, and disjunctions at the top of the syntax tree (num at(·, 0, op)) is markedly different. A + has significantly more conjunctions at the top of the syntax tree than A − , while B + has significantly fewer than B − . Conversely, A + has significantly fewer disjunctions at the top of the syntax tree than A − , while B + has significantly more than DISPLAYFORM3 The mean number of satisfying truth-value assignments (sat(·)) is also markedly different: A + is true in on average 3.7 truth-value assignments (i.e. it is a very specific formula which is only true under very particular circumstances), while A − is true in 10.3 truth-value assignments (i.e. it is true in a wider range of circumstances). We can use these statistics to develop simple heuristic baselines that will be unreasonably effective on the dataset described above: we can estimate whether A B by comparing the lengths of A and B, or by looking at the number of variables in B that do not appear in A, or by looking at the topmost connective in A and B. In order to satisfy our requirements above, we took a different approach to dataset generation. In order to ensure that there are no crude statistical measurements that can detect differences between D + and D − , we change the generation procedure so that every formula appears in both D + and D − . We sample 4-tuples of formulas FIG1 , B 2 ) such that: DISPLAYFORM4 Here, each of the four formulas appears in one positive entailment and one negative entailment * * .Using this alternative approach, we are able to satisfy the requirements above. By construction , the mean length, number of operators at a certain level in the syntax tree, and the number of satisfying truth-value assignments is exactly the same for D + and D − . See Table 4 . DISPLAYFORM5 (r → c) → ((r → v) ∨ p) * * One consequence of this method is that it rules out A1 from being impossible (if it was impossible, we would not have A1 B2) and B1 from being a tautology (if it was a tautology, we would not have A2 B1).
    """

    text42 = "For exercise, Tesla walked between 8 to 10 miles per day. He squished his toes one hundred times for each foot every night, saying that it stimulated his brain cells. What did he do to his feet at night?"

    print(r.compress_prompt(text42))
    r.graph_processor.show_graph()
    print(r.get_ranked_phrases())

    print("RESULTS", r.print_results())
