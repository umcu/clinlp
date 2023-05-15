import json

import spacy

from clinlp.tokenizer import make_tokenizer


class TestTokenizerRegression:
    def test_tokenize_cases(self):
        tokenizer = make_tokenizer(spacy.blank("nl"))

        with open("tests/data/tokenizer_cases.json", "rb") as file:
            data = json.load(file)["data"]

        for example in data:
            assert [token.text for token in tokenizer(example["text"])] == example["tokens"]
