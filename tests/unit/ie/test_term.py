import pytest

from clinlp.ie import Term


class TestTerm:
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

    def test_defaults(self):
        # Act
        defaults = Term.defaults()

        # Assert
        assert defaults == {
            "attr": "TEXT",
            "proximity": 0,
            "fuzzy": 0,
            "fuzzy_min_len": 0,
            "pseudo": False,
        }

    def test_fields_set(self):
        # Arrange
        term = Term(phrase="Diabetes", fuzzy=1)

        # Act
        fields_set = term.fields_set

        # Assert
        assert fields_set == {"phrase", "fuzzy"}

    def test_override_non_set_fields(self):
        # Arrange
        term = Term(phrase="Diabetes", fuzzy=1)
        override_args = {"fuzzy": 2, "fuzzy_min_len": 5}

        # Act
        term = term.override_non_set_fields(override_args)

        # Assert
        assert term.phrase == "Diabetes"
        assert term.fuzzy == 1
        assert term.fuzzy_min_len == 5

    def test_to_spacy_pattern(self, nlp):
        # Arrange
        t = Term(phrase="diabetes", attr="NORM")

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [{"NORM": "diabetes"}]

    def test_to_spacy_pattern_proximity(self, nlp):
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

    def test_to_spacy_pattern_fuzzy(self, nlp):
        # Arrange
        t = Term(phrase="diabetes", fuzzy=3)

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [{"TEXT": {"FUZZY3": "diabetes"}}]

    def test_to_spacy_pattern_fuzzy_min_len(self, nlp):
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

    def test_to_spacy_pattern_pseudo(self, nlp):
        # Arrange
        t = Term(phrase="diabetes", pseudo=True)

        # Act
        spacy_pattern = t.to_spacy_pattern(nlp)

        # Assert
        assert spacy_pattern == [{"TEXT": "diabetes"}]
