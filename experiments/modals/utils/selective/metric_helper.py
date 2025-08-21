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
