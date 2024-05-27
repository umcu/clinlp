import pytest

from clinlp.ie import Term


class TestTerm:
    def test_spacy_pattern_simple(self, nlp):
        # Arrange
        t = Term(phrase="diabetes", attr="NORM")

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [{"NORM": "diabetes"}]

    def test_spacy_pattern_proximity(self, nlp):
        # Arrange
        t = Term("kluts kwijt", proximity=1)

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [
            {"TEXT": "kluts"},
            {"OP": "?"},
            {"TEXT": "kwijt"},
        ]

    def test_spacy_pattern_fuzzy(self, nlp):
        # Arrange
        t = Term(phrase="diabetes", fuzzy=3)

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [{"TEXT": {"FUZZY3": "diabetes"}}]

    def test_spacy_pattern_fuzzy_min_len(self, nlp):
        # Arrange
        t = Term(phrase="bloeding graad iv", fuzzy=1, fuzzy_min_len=6)

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [
            {"TEXT": {"FUZZY1": "bloeding"}},
            {"TEXT": "graad"},
            {"TEXT": "iv"},
        ]

    def test_spacy_pattern_pseudo(self, nlp):
        # Arrange
        t = Term(phrase="diabetes", pseudo=True)

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [{"TEXT": "diabetes"}]

    def test_term_from_dict(self):
        # Arrange
        term_dict = {
            "phrase": "Diabetes",
            "fuzzy": 1,
        }

        # Act
        t = Term(**term_dict)

        # Assert
        assert t.phrase == "Diabetes"
        assert t.fuzzy == 1

    def test_term_from_dict_with_extra_items(self):
        # Arrange
        term_dict = {
            "phrase": "Diabetes",
            "fuzzy": 1,
            "comment": "This term refers to diabetes",
        }

        # Act
        t = Term(**term_dict)

        # Assert
        assert t.phrase == "Diabetes"
        assert t.fuzzy == 1

        with pytest.raises(AttributeError):
            _ = t.comment
