import pytest
from spacy import Vocab
from spacy.tokens import Doc

from clinlp import Normalizer


# Arrange
@pytest.fixture
def mock_doc():
    return Doc(Vocab(), words=["Patiënt", "250", "µg", "toedienen"])


class TestNormalizer:
    @pytest.mark.parametrize(
        ("input_text", "expected_lowercased_text"),
        [
            ("test", "test"),
            ("Test", "test"),
            ("TEST", "test"),
            ("ß", "ss"),
            ("µg", "μg"),
        ],
    )
    def test_lowercase(self, input_text, expected_lowercased_text):
        # Arrange
        n = Normalizer()

        # Act
        lowercased = n._lowercase(input_text)

        # Assert
        assert lowercased == expected_lowercased_text

    @pytest.mark.parametrize(
        ("input_char", "expected_non_ascii_char"),
        [
            ("a", "a"),
            ("à", "a"),
            ("e", "e"),
            ("é", "e"),
            ("ê", "e"),
            ("ë", "e"),
            ("ē", "e"),
            (" ", " "),
            ("\n", "\n"),
            ("µ", "µ"),
            ("²", "²"),
            ("1", "1"),
        ],
    )
    def test_map_non_ascii_char(self, input_char, expected_non_ascii_char):
        # Arrange
        n = Normalizer()

        # Act
        non_ascii = n._map_non_ascii_char(input_char)

        # Assert
        assert non_ascii == expected_non_ascii_char

    def test_map_non_ascii_char_nonchar(self):
        # Arrange
        n = Normalizer()

        # Assert
        with pytest.raises(
            ValueError, match=r".*Please only use the _map_non_ascii_char.*"
        ):
            # Act
            n._map_non_ascii_char("ab")

    @pytest.mark.parametrize(
        ("input_string", "expected_non_ascii_string"),
        [
            ("abcde", "abcde"),
            ("abcdé", "abcde"),
            ("äbcdé", "abcde"),
            (
                "patiënt heeft 1.6m² lichaamsoppervlak",
                "patient heeft 1.6m² lichaamsoppervlak",
            ),
        ],
    )
    def test_map_non_ascii_string(self, input_string, expected_non_ascii_string):
        # Arrange
        n = Normalizer()

        # Act
        non_ascii = n._map_non_ascii_string(input_string)

        # Assert
        assert non_ascii == expected_non_ascii_string

    def test_call_normalizer_default(self, mock_doc):
        # Arange
        expected_norms = ["patient", "250", "μg", "toedienen"]
        n = Normalizer()

        # Act
        doc = n(mock_doc)

        # Assert
        for original_token, token, expected_norm in zip(
            mock_doc, doc, expected_norms, strict=False
        ):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm

    def test_call_normalizer_disable_lowercase(self, mock_doc):
        # Arange
        expected_norms = ["Patient", "250", "µg", "toedienen"]
        n = Normalizer(lowercase=False)

        # Act
        doc = n(mock_doc)

        # Assert
        for original_token, token, expected_norm in zip(
            mock_doc, doc, expected_norms, strict=False
        ):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm

    def test_call_normalizer_disable_map_non_ascii(self, mock_doc):
        # Arange
        expected_norms = ["patiënt", "250", "μg", "toedienen"]
        n = Normalizer(map_non_ascii=False)

        # Act
        doc = n(mock_doc)

        # Assert
        for original_token, token, expected_norm in zip(
            mock_doc, doc, expected_norms, strict=False
        ):
            assert original_token.text == token.text
            assert token.norm_ == expected_norm
