import json

import pytest

from clinlp import Sentencizer


class MockToken:
    def __init__(self, text: str):
        self.text = text
        self.is_sent_start = False


def get_mock_tokens(texts: list[str]):
    return [MockToken(text) for text in texts]


with open("tests/data/sentencizer_cases.json", "rb") as file:
    examples = json.load(file)["data"]

sentencizer_cases = [
    pytest.param(example["tokens"], example["sentence_starts"], id="sentencizer_case")
    for example in examples
]


@pytest.fixture(scope="module")
def sentencizer():
    return Sentencizer()


class TestClinlpSentencizerRegression:
    @pytest.mark.parametrize("tokens, expected_sentence_starts", sentencizer_cases)
    def test_default_clinlp_sentencizer_examples(
        self, sentencizer, tokens, expected_sentence_starts
    ):
        # Arrange
        tokens = get_mock_tokens(tokens)

        # Act
        sentence_starts = sentencizer._get_sentence_starts(tokens)

        # Assert
        assert sentence_starts == expected_sentence_starts
