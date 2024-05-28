import pytest
import spacy

import clinlp  # noqa: F401
from tests.regression import load_examples

tokenizer_cases = [
    pytest.param(example["text"], example["tokens"], id="tokenizer_case_")
    for example in load_examples("tokenizer_cases.json")
]


# Arrange
@pytest.fixture(scope="class")
def tokenizer():
    return spacy.blank("clinlp").tokenizer


class TestTokenizerRegression:
    @pytest.mark.parametrize("text, expected_tokens", tokenizer_cases)
    def test_regression_tokenizer(self, tokenizer, text, expected_tokens):
        # Act
        tokens = [token.text for token in tokenizer(text)]

        # Assert
        assert tokens == expected_tokens
