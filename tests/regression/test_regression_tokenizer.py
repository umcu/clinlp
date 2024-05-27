import json

import pytest
import spacy

import clinlp  # noqa: F401

with open("tests/data/tokenizer_cases.json", "rb") as file:
    examples = json.load(file)["data"]

tokenizer_cases = [
    pytest.param(example["text"], example["tokens"], id="tokenizer_case_")
    for example in examples
]


# Arrange
@pytest.fixture(scope="class")
def tokenizer():
    return spacy.blank("clinlp").tokenizer


class TestTokenizerRegression:
    @pytest.mark.parametrize("text, expected_tokens", tokenizer_cases)
    def test_tokenize_cases(self, tokenizer, text, expected_tokens):
        # Act
        tokens = [token.text for token in tokenizer(text)]

        # Assert
        assert tokens == expected_tokens
