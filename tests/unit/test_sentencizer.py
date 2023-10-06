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
    def test_make_sentencizer(self):
        _ = Sentencizer()

    def test_clinlp_sentencizer_predict_1(self):
        sentencizer = Sentencizer(sent_end_chars=[], sent_start_punct=[])
        tokens = get_mock_tokens(
            ["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"]
        )
        assert sentencizer._get_sentence_starts(tokens) == [
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
        sentencizer = Sentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(
            ["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"]
        )
        assert sentencizer._get_sentence_starts(tokens) == [
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
        sentencizer = Sentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "."])
        assert sentencizer._get_sentence_starts(tokens) == [
            True,
            False,
            False,
            False,
            False,
            False,
        ]

    def test_clinlp_sentencizer_predict_4(self):
        sentencizer = Sentencizer(sent_end_chars=["\n"], sent_start_punct=["*"])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "*", "opsomming"])
        assert sentencizer._get_sentence_starts(tokens) == [
            True,
            False,
            False,
            False,
            False,
            True,
            False,
        ]

    def test_clinlp_sentencizer_start(self):
        sentencizer = Sentencizer(sent_start_punct=["-", "*"])

        assert sentencizer._token_can_start_sent(MockToken("abcde"))
        assert sentencizer._token_can_start_sent(MockToken("1250mg"))
        assert sentencizer._token_can_start_sent(MockToken("[TAG]"))
        assert sentencizer._token_can_start_sent(MockToken("-"))
        assert sentencizer._token_can_start_sent(MockToken("*"))
        assert not sentencizer._token_can_start_sent(MockToken("+"))

    def test_clinlp_sentencizer_end(self):
        sentencizer = Sentencizer(sent_end_chars=["\n", "."])

        assert sentencizer._token_can_end_sent(MockToken("\n"))
        assert sentencizer._token_can_end_sent(MockToken("."))
        assert not sentencizer._token_can_end_sent(MockToken(" "))
        assert not sentencizer._token_can_end_sent(MockToken("abc"))
        assert not sentencizer._token_can_end_sent(MockToken("1250mg"))
        assert not sentencizer._token_can_end_sent(MockToken(","))

    def test_clinlp_sentencizer_call(self):
        sentencizer = Sentencizer()

        tokens = get_mock_tokens(["Dit", "is", "een", "test"])
        expected_returns = [True, False, False, False]

        with patch.object(
            sentencizer, "_get_sentence_starts", return_value=expected_returns
        ):
            sentencizer(tokens)

            for token, expected_return in zip(tokens, expected_returns):
                assert token.is_sent_start == expected_return
