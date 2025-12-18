import importlib.resources


def read_stop_words():
    """
    Reads stopwords from a file and returns them as a set.

    :param file_path: Path to the stopwords file (default: 'stopwords.txt').
    :return: A set of stopwords.
    """
    try:
        with importlib.resources.files("axcer.constants").joinpath("stop_words.txt").open("r") as f:
            stop_words = set(line.strip() for line in f if line.strip())
        return stop_words
    except Exception as e:
        print(f"Error reading stopwords: {e}")
        return set()
