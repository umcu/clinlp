import pytest
from spacy import Vocab
from spacy.tokens import Doc

from clinlp import Normalizer


@pytest.fixture
def mock_doc():
    return Doc(Vocab(), words=["Patiënt", "250", "µg", "toedienen"])


class TestNormalizer:
    def test_lowercase(self):
        # Arrange
        n = Normalizer()

        # Act & Assert
        assert n._lowercase("test") == "test"
        assert n._lowercase("Test") == "test"
        assert n._lowercase("TEST") == "test"

    def test_map_non_ascii_char(self):
        # Arrange
        n = Normalizer()

        # Act & Assert
        assert n._map_non_ascii_char("a") == "a"
        assert n._map_non_ascii_char("à") == "a"
        assert n._map_non_ascii_char("e") == "e"
        assert n._map_non_ascii_char("é") == "e"
        assert n._map_non_ascii_char("ê") == "e"
        assert n._map_non_ascii_char("ë") == "e"
        assert n._map_non_ascii_char("ē") == "e"
        assert n._map_non_ascii_char(" ") == " "
        assert n._map_non_ascii_char("\n") == "\n"
        assert n._map_non_ascii_char("µ") == "µ"
        assert n._map_non_ascii_char("²") == "²"
        assert n._map_non_ascii_char("1") == "1"

    def test_map_non_ascii_char_nonchar(self):
        # Arrange
        n = Normalizer()

        # Act & Assert
        with pytest.raises(ValueError):
            n._map_non_ascii_char("ab")

    def test_map_non_ascii_string(self):
        # Arrange
        n = Normalizer()

        # Act & Assert
        assert n._map_non_ascii_string("abcde") == "abcde"
        assert n._map_non_ascii_string("abcdé") == "abcde"
        assert n._map_non_ascii_string("äbcdé") == "abcde"
        assert (
            n._map_non_ascii_string("patiënt heeft 1.6m² lichaamsoppervlak")
            == "patient heeft 1.6m² lichaamsoppervlak"
        )

    def test_call_normalizer_default(self, mock_doc):
        # Arange
        expected_norms = ["patient", "250", "µg", "toedienen"]
        normalizer = Normalizer()

        # Act
        doc = normalizer(mock_doc)

        # Assert
        for original_token, token, expected_norm in zip(mock_doc, doc, expected_norms):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm

    def test_call_normalizer_disable_lowercase(self, mock_doc):
        # Arange
        expected_norms = ["Patient", "250", "µg", "toedienen"]
        normalizer = Normalizer(lowercase=False)

        # Act
        doc = normalizer(mock_doc)

        # Assert
        for original_token, token, expected_norm in zip(mock_doc, doc, expected_norms):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm

    def test_call_normalizer_disable_map_non_ascii(self, mock_doc):
        # Arange
        expected_norms = ["patiënt", "250", "µg", "toedienen"]
        normalizer = Normalizer(map_non_ascii=False)

        # Act
        doc = normalizer(mock_doc)

        # Assert
        for original_token, token, expected_norm in zip(mock_doc, doc, expected_norms):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm
