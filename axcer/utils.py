def preprocess_text(text):
    punctuation = []
    words = []
    for word in text.split():
        cleaned_word = word.strip("?!,.")
        if cleaned_word:
            words.append(cleaned_word)
    return words
