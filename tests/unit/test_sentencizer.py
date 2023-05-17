from unittest.mock import patch

import pytest
import spacy

import clinlp
from clinlp.sentencizer import ClinlpSentencizer


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
    def test_make_sentencizer(self, nlp):
        _ = ClinlpSentencizer(nlp, name="_")

    def test_clinlp_sentencizer_predict_1(self, nlp):
        sentencizer = ClinlpSentencizer(nlp, sent_end_chars=[], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, False, False, False]

    def test_clinlp_sentencizer_predict_2(self, nlp):
        sentencizer = ClinlpSentencizer(nlp, sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, True, False, False]

    def test_clinlp_sentencizer_predict_3(self, nlp):
        sentencizer = ClinlpSentencizer(nlp, sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "."])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, False]

    def test_clinlp_sentencizer_predict_4(self, nlp):
        sentencizer = ClinlpSentencizer(nlp, sent_end_chars=["\n"], sent_start_punct=["*"])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "*", "opsomming"])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, True, False]

    def test_clinlp_sentencizer_start(self, nlp):
        sentencizer = ClinlpSentencizer(nlp, sent_start_punct=["-", "*"])

        assert sentencizer.token_can_start_sent(MockToken("abcde"))
        assert sentencizer.token_can_start_sent(MockToken("1250mg"))
        assert sentencizer.token_can_start_sent(MockToken("[TAG]"))
        assert sentencizer.token_can_start_sent(MockToken("-"))
        assert sentencizer.token_can_start_sent(MockToken("*"))
        assert not sentencizer.token_can_start_sent(MockToken("+"))

    def test_clinlp_sentencizer_end(self, nlp):
        sentencizer = ClinlpSentencizer(nlp, sent_end_chars=["\n", "."])

        assert sentencizer.token_can_end_sent(MockToken("\n"))
        assert sentencizer.token_can_end_sent(MockToken("."))
        assert not sentencizer.token_can_end_sent(MockToken(" "))
        assert not sentencizer.token_can_end_sent(MockToken("abc"))
        assert not sentencizer.token_can_end_sent(MockToken("1250mg"))
        assert not sentencizer.token_can_end_sent(MockToken(","))

    def test_clinlp_sentencizer_call(self, nlp):
        sentencizer = ClinlpSentencizer(nlp)

        tokens = get_mock_tokens(["Dit", "is", "een", "test"])
        expected_returns = [True, False, False, False]

        with patch.object(sentencizer, "predict", return_value=expected_returns):
            sentencizer(tokens)

            for token, expected_return in zip(tokens, expected_returns):
                assert token.is_sent_start == expected_return
