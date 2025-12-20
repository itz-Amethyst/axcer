import re

APOSTROPHE_RULE = (re.compile(r"\b(\w+)'(\w{3,})\b"), r"\1 ' \2")


def apply_rule(text: str) -> str:
    return APOSTROPHE_RULE[0].sub(APOSTROPHE_RULE[1], text)


def test_apostrophe_rule():
    cases = [
        # Keep compact contractions
        ("it's been cold here", "it's been cold here"),
        ("I'll go there", "I'll go there"),
        ("they're not ready", "they're not ready"),
        ("she'd like to go", "she'd like to go"),
        ("won't you join?", "won't you join?"),

        ("I'msg this is a test", "I ' msg this is a test"),
        ("don'play now", "don ' play now"),
        ("we'run together", "we ' run together"),
        ("someone'example test", "someone ' example test"),

        # Multiple apostrophes
        ("Jack'o'lantern", "Jack'o ' lantern"),
    ]

    for text, expected in cases:
        print(text, expected)
        assert apply_rule(text) == expected
