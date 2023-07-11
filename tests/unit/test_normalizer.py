import pytest
from spacy import Vocab
from spacy.tokens import Doc

from clinlp import Normalizer


@pytest.fixture
def mock_doc():
    return Doc(Vocab(), words=["Patiënt", "250", "µg", "toedienen"])


class TestNormalizer:
    def test_lowercase(self):
        assert Normalizer()._lowercase("test") == "test"
        assert Normalizer()._lowercase("Test") == "test"
        assert Normalizer()._lowercase("TEST") == "test"

    def test_map_non_ascii_char(self):
        assert Normalizer()._map_non_ascii_char("a") == "a"
        assert Normalizer()._map_non_ascii_char("à") == "a"
        assert Normalizer()._map_non_ascii_char("e") == "e"
        assert Normalizer()._map_non_ascii_char("é") == "e"
        assert Normalizer()._map_non_ascii_char("ê") == "e"
        assert Normalizer()._map_non_ascii_char("ë") == "e"
        assert Normalizer()._map_non_ascii_char("ē") == "e"
        assert Normalizer()._map_non_ascii_char(" ") == " "
        assert Normalizer()._map_non_ascii_char("\n") == "\n"
        assert Normalizer()._map_non_ascii_char("µ") == "µ"
        assert Normalizer()._map_non_ascii_char("²") == "²"
        assert Normalizer()._map_non_ascii_char("1") == "1"

    def test_map_non_ascii_char_nonchar(self):
        with pytest.raises(ValueError):
            Normalizer()._map_non_ascii_char("ab")

    def test_map_non_ascii_string(self):
        assert Normalizer()._map_non_ascii_string("abcde") == "abcde"
        assert Normalizer()._map_non_ascii_string("abcdé") == "abcde"
        assert Normalizer()._map_non_ascii_string("äbcdé") == "abcde"
        assert (
            Normalizer()._map_non_ascii_string("patiënt heeft 1.6m² lichaamsoppervlak")
            == "patient heeft 1.6m² lichaamsoppervlak"
        )

    def test_call_normalizer_default(self, mock_doc):
        expected_norms = ["patient", "250", "µg", "toedienen"]
        normalizer = Normalizer()

        doc = normalizer(mock_doc)

        for original_token, token, expected_norm in zip(mock_doc, doc, expected_norms):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm

    def test_call_normalizer_disable_lowercase(self, mock_doc):
        expected_norms = ["Patient", "250", "µg", "toedienen"]
        normalizer = Normalizer(lowercase=False)

        doc = normalizer(mock_doc)

        for original_token, token, expected_norm in zip(mock_doc, doc, expected_norms):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm

    def test_call_normalizer_disable_map_non_ascii(self, mock_doc):
        expected_norms = ["patiënt", "250", "µg", "toedienen"]
        normalizer = Normalizer(map_non_ascii=False)

        doc = normalizer(mock_doc)

        for original_token, token, expected_norm in zip(mock_doc, doc, expected_norms):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm
