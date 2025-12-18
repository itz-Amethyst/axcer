from functools import reduce
import threading
from collections import defaultdict
from itertools import chain

import rustworkx as rx
from joblib import Parallel, delayed, parallel_config

from axcer.constants.question_words import QUESTION_WORDS
from axcer.graph.visualizer import GraphVisualizeMixIn
from axcer.utils import logger

type ProcessCycleReturnType = dict[int, list[str]] | dict[int, str] | int


class GraphProcessor(GraphVisualizeMixIn):
    def __init__(self):
        self.reset()  # Resetting states for each entry process

    def reset(self):
        """Reset all class attributes to their initial state."""
        self.graph = rx.PyDiGraph()
        self.nodes = {}
        self.node_to_color = {}
        self.duplicate_cycles = []  # List to track duplicate cycles
        self.root = []
        self.root_id = []
        self.cycle_hash_map = {}  # Maps hash of cycle path -> list of cycle indices
        self.cycles_dict = defaultdict(list)
        self.cycles = []
        self.correct_cycles = []
        self.results = []
        self.processed_list_weights = []
        self.total_distance_to_interrogative_word: int = 0
        self.number_of_detected_cycle: int = 0

    def add_node(self, label: str) -> int:
        if label not in self.nodes:
            node_id = self.graph.add_node(label)
            self.nodes[label] = node_id
            self.node_to_color[node_id] = None
        return self.nodes[label]

    def add_edge_custom(self, src: int, tgt: int, weight: float) -> None:
        try:
            existing = self.graph.get_edge_data(src, tgt)
        except rx.NoEdgeBetweenNodes:
            existing = None

        if existing is not None:
            new_weight = existing + [weight] if isinstance(existing, list) else [existing, weight]
            self.graph.remove_edge(src, tgt)
            self.graph.add_edge(src, tgt, new_weight)
        else:
            self.graph.add_edge(src, tgt, weight)

    def get_cycle_hash(self, cycle_nodes) -> str:
        """Generate a hash for a cycle path for fast comparison"""
        # Convert node IDs to strings and join with comma for consistent hashing
        path_str = ",".join(map(str, cycle_nodes))
        return path_str

    def build_cycles(self, queue: list[str], phrase: str, index: int) -> None:
        if phrase and queue:
            current_root = self.add_node(phrase)
            self.node_to_color[current_root] = "xkcd:coral"
            self.root.append(phrase)
            self.root_id.append(current_root)
            cycle_path = []

            prev_node = current_root
            for i, item in enumerate(queue, start=1):
                current_node = self.add_node(item)
                if self.node_to_color[current_node] is None:
                    self.node_to_color[current_node] = "xkcd:periwinkle blue"
                weight = float(f"{index}.{i}")
                self.add_edge_custom(prev_node, current_node, weight)

                cycle_path.append(prev_node)
                prev_node = current_node

            last_node = self.add_node(queue[-1])
            weight = float(f"{index}.{len(queue) + 1}")
            cycle_path.append(last_node)
            self.add_edge_custom(last_node, current_root, weight)

            self.cycles_dict[index] = cycle_path

            # Check for duplicates using hash-based approach - O(1) lookup
            cycle_hash = self.get_cycle_hash(cycle_path)

            if cycle_hash in self.cycle_hash_map:
                for existing_index in self.cycle_hash_map[cycle_hash]:
                    duplicate_pair = [existing_index, index]
                    self.duplicate_cycles.append(duplicate_pair)

                self.cycle_hash_map[cycle_hash].append(index)
            else:
                self.cycle_hash_map[cycle_hash] = [index]
            # logger.info("Duplicate Cycles:", self.duplicate_cycles)

    def search_path(self, weight: int) -> list[tuple[int, int]]:
        if weight not in self.cycles_dict:
            return []
        values = self.cycles_dict[weight]

        if not values:
            return []
        pairs = []
        for i in range(len(values) - 1):
            pairs.append((values[i], values[i + 1]))

        pairs.append((values[-1], values[0]))

        return pairs

    def process_paths(self, node_id: int):
        # Get outgoing and incoming edges
        out_edges = self.graph.out_edges(node_id)
        in_edges = self.graph.in_edges(node_id)

        results = []
        processed_weight_indices = set()

        def extract_weight_indices(weight: int | list) -> list[int]:
            """Return a list of integer parts for the weights."""
            if isinstance(weight, list):
                return [int(w) for w in weight]
            return [int(weight)]

        def filter_duplicate_cycles(weight_indices: list[int]) -> list[int]:
            """Filter out indices that appear in the same sublist in duplicate_cycles."""
            if not isinstance(weight_indices, list):
                return [weight_indices]

            # Create a map from index to identify which duplicate cycle it belongs to
            index_to_cycle = {}
            for i, cycle in enumerate(self.duplicate_cycles):
                for idx in cycle:
                    index_to_cycle[idx] = i

            cycle_groups = {}
            for idx in weight_indices:
                if idx in index_to_cycle:
                    cycle_num = index_to_cycle[idx]
                    if cycle_num not in cycle_groups:
                        cycle_groups[cycle_num] = []
                    cycle_groups[cycle_num].append(idx)

            filtered_indices = []
            for idx in weight_indices:
                if idx not in index_to_cycle:
                    filtered_indices.append(idx)
                    continue

                cycle_num = index_to_cycle[idx]
                if cycle_groups.get(cycle_num):
                    filtered_indices.append(cycle_groups[cycle_num][0])
                    # Remove this cycle group in order to not process it again
                    cycle_groups.pop(cycle_num)

            return filtered_indices

        for out_edge in out_edges:
            source, target, weight = out_edge
            weight_indices = extract_weight_indices(weight)

            if isinstance(weight_indices, list):
                # Filter out duplicates from the same cycle
                indices_to_process = filter_duplicate_cycles(weight_indices)
            else:
                indices_to_process = weight_indices

            for weight_index in indices_to_process:
                if weight_index in processed_weight_indices:
                    continue
                processed_weight_indices.add(weight_index)

                matching_in_edges = []
                for in_edge in in_edges:
                    in_source, in_target, in_weight = in_edge
                    in_weight_indices = extract_weight_indices(in_weight)
                    if weight_index in in_weight_indices and self.cycles_dict[weight_index][0] == in_target:
                        matching_in_edges.append((in_source, in_target, weight_index))
                        break

                # If matching in_edges are found, select the one with highest fractional number
                if matching_in_edges:
                    # To devoke from list
                    matched_cycle = matching_in_edges[0]
                    best_in_edge = matched_cycle

                    # Check if the weight (third element) is a list
                    if isinstance(best_in_edge[2], list):
                        for weight_value in best_in_edge[2]:
                            res = self.search_path(int(weight_value))
                            results.append(res)
                    else:
                        results.append(self.search_path(int(best_in_edge[2])))

        return results

    def find_common_prefixes(self, data):
        """
        Find number prefixes that appear in all sublists.
        Args:
            data: List of lists containing numeric values
        Returns:
            Set of common prefixes across all sublists
        """

        if not data or all(not sublist for sublist in data):
            return set()

        sets = [set(sublist) for sublist in data]

        if sets:
            common_elements = reduce(set.intersection, sets)
            return common_elements
        return set()

    def process_single_cycle(self, cycle: list, trigger_words: set[str]) -> dict[str, ProcessCycleReturnType]:
        """
        Process a single cycle to extract paths.
        Args:
            cycle (List): List of edges forming a cycle
            trigger_words (Set[str]): Words that trigger path extraction
        Returns:
            Dict[int, List[List[str]]]: Dictionary of extracted paths indexed by edge weights
        """

        if len(cycle) <= 1:
            return {}

        node_path = [edge[0] for edge in cycle]

        trigger_set = frozenset(trigger_words)
        results = {}

        all_weights = []
        list_weights = []
        updated_node_to_color = {}
        saved_words = []
        single_weight = None
        saving_mode = threading.Event()
        if self.processed_list_weights and list(node_path) in list(self.processed_list_weights):
            return {}

        for i in range(len(node_path)):
            current_node = node_path[i]
            next_node = node_path[(i + 1) % len(node_path)]
            current_word = self.graph.get_node_data(current_node)

            # Get weight for every node
            try:
                if not single_weight and not list_weights:
                    weight = self.graph.get_edge_data(current_node, next_node)
                    all_weights.append(weight)

                    if not isinstance(weight, list) and single_weight is None:
                        single_weight = weight
                    elif isinstance(weight, list):
                        list_weight = [int(float(w)) for w in weight]
                        list_weights.append(list_weight)

            except Exception as e:
                logger.error(f"Edge processing error: {e}")
                all_weights.append(None)

            if current_word.lower() in trigger_set or saving_mode.is_set():
                saving_mode.set()
                saved_words.append(current_word)

        for word in saved_words:
            node_id = self.add_node(word)
            updated_node_to_color[node_id] = "xkcd:minty green"

        if saved_words:
            self.total_distance_to_interrogative_word += len(saved_words)
            self.number_of_detected_cycle += 1

            if single_weight is not None:
                key = int(float(single_weight))
                results[key] = saved_words
                return {
                    "paths": results,
                    "node_colors": updated_node_to_color,
                    "total_distance_to_interrogative_word": len(saved_words),
                    "number_of_detected_cycle": 1,
                }

            if list_weights:
                common_prefixes = self.find_common_prefixes(list_weights)

                if common_prefixes:
                    for prefix in common_prefixes:
                        key = int(prefix)
                        results[key] = saved_words.copy()
                        # to save weights which to not let them re run again
                    self.processed_list_weights.append(node_path)
                    return {
                        "paths": results,
                        "node_colors": updated_node_to_color,
                        "total_distance_to_interrogative_word": len(saved_words),
                        "number_of_detected_cycle": 1,
                    }

        return {"paths": results, "node_colors": updated_node_to_color}

    def extract_paths_from_cycles(
        self,
        cycles: list[list],
        trigger_words: set[str] | None = None,
    ) -> dict[int, list[list[str]]]:
        """
        Extract paths from cycles using parallel processing.

        Args:
            cycles (List[List]): List of cycles to process
            trigger_words (List[str], optional): Words that trigger path extraction

        Returns:
            Dict[int, List[List[str]]]: Aggregated results from all cycles
        """
        if trigger_words is None:
            trigger_words = QUESTION_WORDS

        final_results = {}

        # will only run on 1 process
        with parallel_config(
            # backend="threading",  # force process backend
            # backend="multiprocessing",  # force process backend
            backend="threading",  # force process backend
            # n_jobs=-1,  # use all CPUs
            n_jobs=1,
            verbose=0,  # show what's happening
            prefer="threads",  # soft hint to use threads
        ):
            results = Parallel()(delayed(self.process_single_cycle)(cycle, trigger_words) for cycle in cycles)
            for result in results:
                try:
                    cycle_results = result.get("paths", {}) or {}
                    node_colors = result.get("node_colors", {}) or {}
                    self.total_distance_to_interrogative_word += result.get("total_distance_to_interrogative_word", 0)
                    self.number_of_detected_cycle += result.get("number_of_detected_cycle", 0)

                    for key, paths in cycle_results.items():
                        if paths:
                            final_results.setdefault(key, []).extend(paths)

                    if node_colors:
                        self.node_to_color.update(node_colors)

                except Exception as e:
                    logger.error(f"Cycle processing error: {e}")

        return final_results

    def check_cycle_validation(self, start_node, cycle) -> bool:
        """
        Validates if the starting node matches the first node in the cycle.

        Args:
            start_node: The node we started from (i)
            cycle: The cycle found by rx.digraph_find_cycle

        Returns:
            bool: True if the cycle starts with the start_node, False otherwise
        """
        if not cycle:
            return False

        first_edge = cycle[0]

        first_node_in_cycle = first_edge[0]

        return first_node_in_cycle == start_node

    def correct_process(self, root_id: int) -> list[list[tuple[int, int]]]:
        """
        Process paths starting from a given root node and add them to correct_cycles.

        Args:
            root_id (int): The node ID to start path processing from

        Returns:
            list: The updated correct_cycles list after processing
        """
        try:
            result = self.process_paths(root_id)

            if result:
                if isinstance(result, list):
                    self.correct_cycles.extend(result)
                else:
                    self.correct_cycles.append(result)

            else:
                logger.warning(f"No valid paths found from node {root_id}")

            return self.correct_cycles
        except Exception as e:
            logger.error(f"Error processing paths from node {root_id}: {e}")
            return self.correct_cycles

    def extract_interrogative_phrases(self):
        """
        Extract interrogative phrases from the graph and reset the state for next use.

        Returns:
            dict: The extracted paths from cycles
        """
        index_counts = {}
        for idx in self.root_id:
            index_counts[idx] = index_counts.get(idx, 0) + 1

        with parallel_config(
            backend="threading",  # force process backend
            n_jobs=1,
            verbose=0,
            prefer="threads",
        ):
            unique_list = []
            duplicate_list = []
            for idx, count in index_counts.items():
                if count == 1:
                    unique_list.append(idx)
                else:
                    duplicate_list.append(idx)

            Parallel()(delayed(self.process_unique)(idx) for idx in unique_list)
            Parallel()(delayed(self.process_duplicate)(idx) for idx in duplicate_list)
            merged = list(chain(self.cycles, self.correct_cycles))

            result = self.extract_paths_from_cycles(merged)

        return result

    def process_unique(self, idx):
        cycle = rx.digraph_find_cycle(self.graph, idx)
        start = cycle[0][0]
        end = cycle[-1][0]

        all_paths = rx.all_simple_paths(self.graph, start, end, cutoff=5)
        if self.check_cycle_validation(idx, cycle) and len(all_paths) == 1:
            self.cycles.append(cycle)
        else:
            self.correct_process(idx)

    def process_duplicate(self, idx: int) -> None:
        self.correct_process(idx)
