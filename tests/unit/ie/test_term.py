import pytest
import spacy

from clinlp.ie import Term


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


class TestTerm:
    def test_create_term(self):
        assert Term(phrase="diabetes")
        assert Term(phrase="diabetes", attr="NORM").attr == "NORM"
        assert Term(phrase="diabetes", proximity=1).proximity == 1
        assert Term(phrase="diabetes", fuzzy=1).fuzzy == 1
        assert Term(phrase="diabetes", fuzzy_min_len=1).fuzzy_min_len == 1
        assert Term(phrase="diabetes", pseudo=True).pseudo

    def test_spacy_pattern_simple(self, nlp):
        t = Term(phrase="diabetes", attr="NORM")

        assert t.to_spacy_pattern(nlp) == [{"NORM": "diabetes"}]

    def test_spacy_pattern_proximity(self, nlp):
        t = Term("kluts kwijt", proximity=1)

        assert t.to_spacy_pattern(nlp) == [
            {"TEXT": "kluts"},
            {"OP": "?"},
            {"TEXT": "kwijt"},
        ]

    def test_spacy_pattern_fuzzy(self, nlp):
        t = Term(phrase="diabetes", fuzzy=3)

        assert t.to_spacy_pattern(nlp) == [{"TEXT": {"FUZZY3": "diabetes"}}]

    def test_spacy_pattern_fuzzy_min_len(self, nlp):
        t = Term(phrase="bloeding graad iv", fuzzy=1, fuzzy_min_len=6)

        assert t.to_spacy_pattern(nlp) == [
            {"TEXT": {"FUZZY1": "bloeding"}},
            {"TEXT": "graad"},
            {"TEXT": "iv"},
        ]

    def test_spacy_pattern_pseudo(self, nlp):
        t = Term(phrase="diabetes", pseudo=True)

        assert t.to_spacy_pattern(nlp) == [{"TEXT": "diabetes"}]

    def test_term_from_dict(self):
        t = Term(
            **{
                "phrase": "Diabetes",
                "fuzzy": 1,
            }
        )

        assert t.phrase == "Diabetes"
        assert t.fuzzy == 1

    def test_term_from_dict_with_extra_items(self):
        t = Term(
            **{
                "phrase": "Diabetes",
                "fuzzy": 1,
                "comment": "This term refers to diabetes",
            }
        )

        assert t.phrase == "Diabetes"
        assert t.fuzzy == 1

        with pytest.raises(AttributeError):
            _ = t.comment
