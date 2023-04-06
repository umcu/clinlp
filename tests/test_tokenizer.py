import json

import spacy

from clinlp.tokenize import create_tokenizer


class TestTokenizer:
    def test_create_tokenizer(self):
        _ = create_tokenizer(spacy.blank("nl"))

    def test_tokenize_cases(self):
        tokenizer = create_tokenizer(spacy.blank("nl"))

        with open("tests/data/tokenizer_cases.json", "rb") as file:
            data = json.load(file)["data"]

        for example in data:
            assert [token.text for token in tokenizer(example["text"])] == example["tokens"]
