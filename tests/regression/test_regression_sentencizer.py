import pytest

from clinlp import Sentencizer
from tests.conftest import get_mock_tokens
from tests.regression import load_examples

sentencizer_cases = [
    pytest.param(example["tokens"], example["sentence_starts"], id="sentencizer_case_")
    for example in load_examples("data/sentencizer_cases.json")
]


# Arrange
@pytest.fixture(scope="class")
def sentencizer():
    return Sentencizer()


class TestClinlpSentencizerRegression:
    @pytest.mark.parametrize(("tokens", "expected_sentence_starts"), sentencizer_cases)
    def test_regression_sentencizer(
        self, sentencizer, tokens, expected_sentence_starts
    ):
        # Arrange
        tokens = get_mock_tokens(tokens)

        # Act
        sentence_starts = sentencizer._compute_sentence_starts(tokens)

        # Assert
        assert sentence_starts == expected_sentence_starts
