import time
import sys


from selective_context import SelectiveContext


def main(input_text: str, reduce_ratio: float = 0.5):
    """
    Compresses `input_text` using Selective Context API while measuring elapsed time.
    Outputs:
      - number of words before/after,
      - compression rate,
      - time taken (sec),
      - compressed context,
      - dropped content (optional).
    """

    # 1) Initialize the compressor
    print(f"📦 Initializing SelectiveContext(model_type='gpt2', lang='en')")
    sc = SelectiveContext(model_type="gpt2", lang="en")

    # 2) Run compression with timer
    t0 = time.perf_counter()
    context_kept, reduced_content = sc(input_text, reduce_ratio=reduce_ratio)
    t1 = time.perf_counter()

    # 3) Compute metrics
    original_len = len(input_text.split())
    kept_len = len(context_kept.split())
    dropped_len = len(reduced_content.split()) if reduced_content else (original_len - kept_len)
    rate = kept_len / original_len if original_len else 1.0
    elapsed = t1 - t0

    # 4) Print results
    print("\n=== Compression Results ===")
    print(f"Original words: {original_len}")
    print(f"Kept words:     {kept_len}")
    print(f"Dropped words:  {dropped_len}")
    print(f"Compression ratio: {rate:.3f} ({kept_len / original_len:.1%} of original)")
    print(f"🕒 Time elapsed: {elapsed:.3f} seconds")
    print("\n--- Compressed context (kept) ---")
    print(context_kept[:500] + ("…" if len(context_kept) > 500 else ""))
    print("\n--- Optional dropped content ---")
    print(reduced_content[:500] + ("…" if reduced_content and len(reduced_content) > 500 else ""))


if __name__ == "__main__":
    txt = "Mosquito fish found on the islands of the Bahamas live in various isolated freshwater ponds that were once a single body of water. When several male and female mosquito fish are taken from two isolated ponds and placed into a single pond, the breeding preference of each mosquito fish is for fish from its own original pond. Which of these most likely resulted in this breeding preference? Options: A. Availability of food influenced the breeding preferences of the fish., B. Competition for a suitable mate influenced the breeding preferences., C. Predators in the pond influenced the breeding preferences of the fish., D. Speciation due to reproductive isolation influenced the breeding preferences."
    txt = "How do I ready a guinea pig cage for it's new occupants?, Options: A. Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish., B. Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."

    main(txt, reduce_ratio=0.5)
