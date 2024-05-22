import json

import spacy

import clinlp  # noqa: F401


class TestTokenizerRegression:
    def test_tokenize_cases(self):
        # Arrange
        tokenizer = spacy.blank("clinlp").tokenizer

        with open("tests/data/tokenizer_cases.json", "rb") as file:
            data = json.load(file)["data"]

        for example in data:
            # Act
            tokens = [token.text for token in tokenizer(example["text"])]

            # Assert
            assert tokens == example["tokens"]
