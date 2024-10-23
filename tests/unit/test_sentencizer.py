from unittest.mock import patch

import pytest
from tests.conftest import MockToken, get_mock_tokens

from clinlp import Sentencizer


class TestUnitClinlpSentencizer:
    @pytest.mark.parametrize(
        ("text", "expected_can_start_sent"),
        [
            ("abcde", True),
            ("1250mg", True),
            ("[TAG]", True),
            ("-", True),
            ("*", True),
            ("+", False),
        ],
    )
    def test_sentencizer_can_start_sent(self, text, expected_can_start_sent):
        # Arrange
        s = Sentencizer(sent_start_punct=["-", "*"])

        # Act
        can_start_sent = s._token_can_start_sent(MockToken(text))

        # Assert
        assert can_start_sent == expected_can_start_sent

    @pytest.mark.parametrize(
        ("text", "expected_can_end_sent"),
        [
            ("\n", True),
            (".", True),
            (" ", False),
            ("abc", False),
            ("1250mg", False),
            (",", False),
        ],
    )
    def test_sentencizer_can_end_sent(self, text, expected_can_end_sent):
        # Arrange
        s = Sentencizer(sent_end_chars=["\n", "."])

        # Act
        can_end_sent = s._token_can_end_sent(MockToken(text))

        # Assert
        assert can_end_sent == expected_can_end_sent

    def test_sentencizer_compute_sentence_starts_1(self):
        # Arrange
        s = Sentencizer(sent_end_chars=[], sent_start_punct=[])
        tokens = get_mock_tokens(
            ["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"]
        )

        # Act
        sentence_starts = s._compute_sentence_starts(tokens)

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

    def test_sentencizer_compute_sentence_starts_2(self):
        # Arrange
        s = Sentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(
            ["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"]
        )

        # Act
        sentence_starts = s._compute_sentence_starts(tokens)

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

    def test_sentencizer_compute_sentence_starts_3(self):
        # Arrange
        s = Sentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "."])

        # Act
        sentence_starts = s._compute_sentence_starts(tokens)

        # Assert
        assert sentence_starts == [
            True,
            False,
            False,
            False,
            False,
            False,
        ]

    def test_sentencizer_compute_sentence_starts_4(self):
        # Arrange
        s = Sentencizer(sent_end_chars=["\n"], sent_start_punct=["*"])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "*", "opsomming"])

        # Act
        sentence_starts = s._compute_sentence_starts(tokens)

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

    def test_sentencizer_call(self):
        # Arrange
        s = Sentencizer()
        tokens = get_mock_tokens(["Dit", "is", "een", "test"])
        expected_returns = [True, False, False, False]

        # Act
        with patch.object(s, "_compute_sentence_starts", lambda _: expected_returns):
            s(tokens)

        # Assert
        for token, expected_return in zip(tokens, expected_returns, strict=False):
            assert token.is_sent_start == expected_return
