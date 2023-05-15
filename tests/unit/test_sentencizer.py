from unittest.mock import patch

import spacy

from clinlp.sentencizer import ClinlpSentencizer, make_sentencizer


class MockToken:
    def __init__(self, text: str):
        self.text = text
        self.is_sent_start = False


def get_mock_tokens(texts: list[str]):
    return [MockToken(text) for text in texts]


class TestUnitClinlpSentencizer:
    def test_make_sentencizer(self):
        _ = make_sentencizer(spacy.blank("nl"), name="_")

    def test_clinlp_sentencizer_predict_1(self):
        sentencizer = ClinlpSentencizer(sent_end_chars=[], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, False, False, False]

    def test_clinlp_sentencizer_predict_2(self):
        sentencizer = ClinlpSentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "met", "twee", "zinnen"])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, True, False, False]

    def test_clinlp_sentencizer_predict_3(self):
        sentencizer = ClinlpSentencizer(sent_end_chars=["\n"], sent_start_punct=[])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "."])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, False]

    def test_clinlp_sentencizer_predict_4(self):
        sentencizer = ClinlpSentencizer(sent_end_chars=["\n"], sent_start_punct=["*"])
        tokens = get_mock_tokens(["dit", "is", "een", "test", "\n", "*", "opsomming"])
        assert sentencizer.predict(tokens) == [True, False, False, False, False, True, False]

    def test_clinlp_sentencizer_start(self):
        sentencizer = ClinlpSentencizer(sent_start_punct=["-", "*"])

        assert sentencizer.token_can_start_sent(MockToken("abcde"))
        assert sentencizer.token_can_start_sent(MockToken("1250mg"))
        assert sentencizer.token_can_start_sent(MockToken("[TAG]"))
        assert sentencizer.token_can_start_sent(MockToken("-"))
        assert sentencizer.token_can_start_sent(MockToken("*"))
        assert not sentencizer.token_can_start_sent(MockToken("+"))

    def test_clinlp_sentencizer_end(self):
        sentencizer = ClinlpSentencizer(sent_end_chars=["\n", "."])

        assert sentencizer.token_can_end_sent(MockToken("\n"))
        assert sentencizer.token_can_end_sent(MockToken("."))
        assert not sentencizer.token_can_end_sent(MockToken(" "))
        assert not sentencizer.token_can_end_sent(MockToken("abc"))
        assert not sentencizer.token_can_end_sent(MockToken("1250mg"))
        assert not sentencizer.token_can_end_sent(MockToken(","))

    def test_clinlp_sentencizer_call(self):
        sentencizer = ClinlpSentencizer()

        tokens = get_mock_tokens(["Dit", "is", "een", "test"])
        expected_returns = [True, False, False, False]

        with patch.object(sentencizer, "predict", return_value=expected_returns):
            sentencizer(tokens)

            for token, expected_return in zip(tokens, expected_returns):
                assert token.is_sent_start == expected_return
