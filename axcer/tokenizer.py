import re


DE_STARTING_QUOTES = [
    (re.compile(r"(?<=\S) ([«“‘„]|[`]+) (?=\S)"), r"\1"),
    (re.compile(r"^`"), r'"'),
]

DE_ENDING_QUOTES = [(re.compile(r"\s*([»”’])\s*"), r"\1"), (re.compile(r"\s*''\s*"), r"''"), (re.compile(r"\s*''\s*"), r'"')]

DE_PARENS_BRACKETS = [(re.compile(r"\s*([\]\[\(\)\{\}\<\>])\s*"), r"\1")]
DE_PUNCTUATION = [
    (
        re.compile(r"\s([:,])"),
        r"\1",
    ),
    (re.compile(r"([^\.])\s+(\.)\s+([\]\)}>'\"»”’]*)\s*$"), r"\1\2\3"),
    (re.compile(r"(?<!\d)\b(\w+)\s+\.(?!\d)"), r"\1."),
    # (re.compile(r"\s+([:,])\s+$"), r"\1"),
    (re.compile(r"\s*([:,])\s*"), r"\1"),
    (re.compile(r"\s+(\.{2,})\s+"), r"\1"),
    (re.compile(r"\s*([;#&/\+])\s*"), r"\1"),
    (re.compile(r"(\d+)\s%"), r"\1%"),
    # (re.compile(r"\s*([?!])\s*"), r"\1"),
    (re.compile(r"\s+([?!])\s*"), r"\1 "),
    (re.compile(r"([^'])\s' "), r"\1' "),
]


class RegexTokenizer:
    """
    The Custom Regex-Based tokenizer for preprocessing the text prior sending to compression stage.
    """

    STARTING_QUOTES = [
        (re.compile(r"[«“‘„`]+", re.U), ""),  # remove opening quotes/backticks
        (re.compile(r'^"'), ""),  # remove starting " at beginning of line
        (re.compile(r"(`)"), ""),  # remove stray backticks
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1"),  # remove starting quotes after brackets
    ]

    ENDING_QUOTES = [
        (re.compile(r"[»”’]"), ""),  # remove closing quotes
        (re.compile(r"''"), ""),  # remove two apostrophes
        (re.compile(r'"'), ""),  # remove leftover "
        (re.compile(r"\s+"), " "),  # normalize spaces
    ]

    PUNCTUATION = [
        (re.compile(r"([^\.])(\.)([\]\)}>'\"»”’]*)\s*$"), r"\1 \2 \3 "),
        # To support what. => what .
        (re.compile(r"(?<!\d)\b(\w+)\.(?!\d)"), r"\1 ."),
        (re.compile(r"([:,])([^\d])"), r" \1 \2"),
        (re.compile(r"([:,])$"), r" \1 "),
        (re.compile(r"\.{2,}"), r" \g<0> "),
        (re.compile(r"[;#&/\+]"), r" \g<0> "),
        (re.compile(r"[$%]"), r" \g<0>"),
        (re.compile(r"(?<![@\w])-(?![@\w._])"), r" \g<0> "),
        (re.compile(r"[?!]"), r" \g<0> "),
        # old aposthrophe rule
        # (re.compile(r"([^'])' "), r"\1 ' "),
        (re.compile(r"\b(\w+)'(\w{3,})\b"), r"\1 ' \2"),
        (re.compile(r"[*]"), r" \g<0> "),
    ]

    ID_PATTERN = (re.compile(r"(?<![\w._])(@[\w._-]+)"), r"\g<0>")

    CONTRACTIONS_PATTERN = re.compile(r"(?i)\b(\w+)'([a-z]+)\b")

    # LINK_PATTERN = re.compile(
    #     r"((https?|ftp):/{1,2})?(?<!@)([a-zA-Z0-9-]+\.[a-zA-Z]{2,}|localhost)(:\d+)?[-\w@:%_.+/~#?=&]*",
    # )
    LINK_PATTERN = re.compile(r"((https?|ftp):/{1,2})?(?<!@)([a-zA-Z0-9-]+\.[a-zA-Z]{2,}|localhost)(:\d+)?[-\w@:%_.+/~#?=&]*")

    NUMBER_PATTERN = re.compile(
        rf"\b(?:[A-Za-z]{3, 4}\s)?[\$\€\£\₹\¥\₣]?\s?\d+(\.\d+)?(?:[\$\€\£\₹\¥\₣]|\s?[A-Za-z]{3, 4})?\b|\w+|\S+"
    )

    PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")

    # DOUBLE_DASHES = (re.compile(r"--"), r" -- ")
    DOUBLE_DASHES = (re.compile(r"(--|—)"), r" ")

    def _generate_unique_prefix(self, link: str) -> str:
        """
        Generate a unique prefix for the link using its last 4 characters.
        """
        return link[-4:]

    def _tag_links(self, text: str) -> str:
        """
        Tag all links in the text with a dynamic prefix.
        """

        def replace_link_with_prefix(match):
            link = match.group(0)
            unique_prefix = self._generate_unique_prefix(link)
            return f"<LINK_{unique_prefix}>{link}</LINK_{unique_prefix}>"

        return self.LINK_PATTERN.sub(replace_link_with_prefix, text)

    def _clean_matching_links(self, text) -> str:
        pattern = r"<LINK_([\w?./:=&%#\-ـ]+)>(.+?)</LINK_\1>"

        cleaned_text = re.sub(pattern, r"\2", text)

        return cleaned_text

    def _apply_punctuation(self, tagged_text: str) -> str:
        """
        Apply punctuation rules to non-tagged parts of the text, preserving <LINK> tags.
        """
        result = []
        i = 0

        while i < len(tagged_text):
            if tagged_text[i] == "<":
                if tagged_text.startswith("<LINK_", i):
                    start_tag_end = tagged_text.find(">", i)
                    if start_tag_end == -1:
                        result.append(tagged_text[i:])
                        break

                    identifier = tagged_text[i + len("<LINK_") : start_tag_end]
                    closing_tag = f"</LINK_{identifier}>"

                    close_index = tagged_text.find(closing_tag, start_tag_end + 1)

                    if close_index != -1:
                        block = tagged_text[i : close_index + len(closing_tag)]
                        result.append(block)
                        i = close_index + len(closing_tag)
                    else:
                        result.append(tagged_text[i:])
                        break
                else:
                    result.append(tagged_text[i])
                    i += 1
            else:
                # Apply punctuation rules
                for regexp, substitution in self.PUNCTUATION:
                    non_tagged_segment = regexp.sub(substitution, tagged_text[i:])

                # Append the processed non-tagged segment
                result.append(non_tagged_segment)
                i += 1

        return "".join(result)

    def _apply_punctuation_to_non_links(
        self, text: str, *regex_sub_pairs: tuple[re.Pattern, str] | list[tuple[re.Pattern, str]]
    ) -> str:
        """
        Apply regex substitutions to non-link sections of the text.

        :param text: Input text containing optional <LINK_4WORD>...</LINK_4WORD> tags.
        :param regex_sub_pairs: Regex–substitution pair or list of pairs.
        :return: Text with substitutions applied outside link tags.
        """
        pairs = []
        for item in regex_sub_pairs:
            if isinstance(item, tuple):
                pairs.append(item)
            elif isinstance(item, list):
                pairs.extend(item)

        result = []
        i = 0  # Start at the beginning of the text

        while i < len(text):
            if text.startswith("<LINK_", i):
                start_tag_end = text.find(">", i)
                if start_tag_end == -1:
                    result.append(text[i:])
                    break

                identifier = text[i + len("<LINK_") : start_tag_end]
                closing_tag = f"</LINK_{identifier}>"

                close_index = text.find(closing_tag, start_tag_end + 1)

                if close_index != -1:
                    block = text[i : close_index + len(closing_tag)]
                    result.append(block)
                    i = close_index + len(closing_tag)
                else:
                    result.append(text[i])
                    i += 1

            else:
                next_link = text.find("<LINK_", i)
                if next_link == -1:
                    chunk = text[i:]
                    for regex, sub in pairs:
                        chunk = regex.sub(sub, chunk)
                    result.append(chunk)
                    break
                else:
                    chunk = text[i:next_link]
                    for regex, sub in pairs:
                        chunk = regex.sub(sub, chunk)
                    result.append(chunk)
                    i = next_link

        return "".join(result)

    def tokenize(
        self,
        text: str,
    ) -> list[str]:
        """
        Tokenize the input text into a list of strings.

        :param text: Input text to tokenize.
        :return: List of tokenized tokens.
        """

        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        text = self.CONTRACTIONS_PATTERN.sub(r"\1'\2", text)

        # Handles parentheses
        regexp, substitution = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)

        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # Add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        text = self._tag_links(text)
        # print("Text after tagging links:", text)

        text = self._apply_punctuation_to_non_links(text, self.PUNCTUATION, self.ID_PATTERN)

        # print("Tokenized text", text)

        text = self._clean_matching_links(text)

        # return text.lower().split()
        return text.split()

    def detokenize(self, tokens: list[str]) -> str:
        text = " ".join(tokens)

        # Apply all regex rules in sequence
        for pattern, repl in DE_STARTING_QUOTES + DE_ENDING_QUOTES + DE_PARENS_BRACKETS + DE_PUNCTUATION:
            text = pattern.sub(repl, text)

        return text.strip()
