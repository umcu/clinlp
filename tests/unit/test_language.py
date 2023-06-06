import warnings

import spacy

import clinlp


class TestUnitCreateModel:
    def test_create_model(self):
        _ = spacy.blank("clinlp")

    def test_apply_model(self):
        nlp = spacy.blank("clinlp")
        nlp("dit is een test")

    def test_version(self):
        nlp = spacy.blank("clinlp")
        assert "clinlp_version" in nlp.meta

    def test_load_wrong_version(self):
        with warnings.catch_warnings(record=True) as w:
            _ = spacy.load("tests/data/test_spacy_model")

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
