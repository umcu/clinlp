import pytest

from tests.conftest import _make_nlp
from tests.regression import load_examples

tokenizer_cases = [
    pytest.param(example["text"], example["tokens"], id="tokenizer_case_")
    for example in load_examples("data/tokenizer_cases.json")
]


@pytest.fixture(scope="class")
def nlp():
    return _make_nlp()


# Arrange
@pytest.fixture(scope="class")
def tokenizer(nlp):
    return nlp.tokenizer


class TestTokenizerRegression:
    @pytest.mark.parametrize(("text", "expected_tokens"), tokenizer_cases)
    def test_regression_tokenizer(self, tokenizer, text, expected_tokens):
        # Act
        tokens = [token.text for token in tokenizer(text)]

        # Assert
        assert tokens == expected_tokens
