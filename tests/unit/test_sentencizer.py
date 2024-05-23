from unittest.mock import patch

import pytest
import spacy

from clinlp import Sentencizer


class MockToken:
    def __init__(self, text: str):
        self.text = text
        self.is_sent_start = False


def get_mock_tokens(texts: list[str]):
    return [MockToken(text) for text in texts]


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


class TestUnitClinlpSentencizer:
    def test_clinlp_sentencizer_predict_1(self):
        # Arrange
        sentencizer = Sentencizer(sent_end_chars=[], sent_start_punct=[])
        tokens = get_mock_tokens(
            ["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"]
        )

        # Act
        sentence_starts = sentencizer._get_sentence_starts(tokens)

        # Assert
        assert sentence_starts == [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

    def test_clinlp_sentencizer_predict_2(self):
        # Arrange
        sentencizer = Sentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(
            ["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"]
        )

        # Act
        sentence_starts = sentencizer._get_sentence_starts(tokens)

        # Assert
        assert sentence_starts == [
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ]

    def test_clinlp_sentencizer_predict_3(self):
        # Arrange
        sentencizer = Sentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "."])

        # Act
        sentence_starts = sentencizer._get_sentence_starts(tokens)

        # Assert
        assert sentence_starts == [
            True,
            False,
            False,
            False,
            False,
            False,
        ]

    def test_clinlp_sentencizer_predict_4(self):
        # Arrange
        sentencizer = Sentencizer(sent_end_chars=["\n"], sent_start_punct=["*"])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "*", "opsomming"])

        # Act
        sentence_starts = sentencizer._get_sentence_starts(tokens)

        # Assert
        assert sentence_starts == [
            True,
            False,
            False,
            False,
            False,
            True,
            False,
        ]

    @pytest.mark.parametrize(
        "text, expected_can_start_sent",
        [
            ("abcde", True),
            ("1250mg", True),
            ("[TAG]", True),
            ("-", True),
            ("*", True),
            ("+", False),
        ],
    )
    def test_clinlp_sentencizer_start(self, text, expected_can_start_sent):
        # Arrange
        sentencizer = Sentencizer(sent_start_punct=["-", "*"])

        # Act
        can_start_sent = sentencizer._token_can_start_sent(MockToken(text))

        # Assert
        assert can_start_sent == expected_can_start_sent

    @pytest.mark.parametrize(
        "text, expected_can_end_sent",
        [
            ("\n", True),
            (".", True),
            (" ", False),
            ("abc", False),
            ("1250mg", False),
            (",", False),
        ],
    )
    def test_clinlp_sentencizer_end(self, text, expected_can_end_sent):
        # Arrange
        sentencizer = Sentencizer(sent_end_chars=["\n", "."])

        # Act
        can_end_sent = sentencizer._token_can_end_sent(MockToken(text))

        # Assert
        assert can_end_sent == expected_can_end_sent

    def test_clinlp_sentencizer_call(self):
        # Arrange
        sentencizer = Sentencizer()
        tokens = get_mock_tokens(["Dit", "is", "een", "test"])
        expected_returns = [True, False, False, False]

        with patch.object(
            sentencizer, "_get_sentence_starts", return_value=expected_returns
        ):
            # Act
            sentencizer(tokens)

            # Assert
            for token, expected_return in zip(tokens, expected_returns):
                assert token.is_sent_start == expected_return
