import spacy

import clinlp


class TestUnitCreateModel:
    def test_create_model(self):
        _ = spacy.blank("clinlp")

    def test_apply_model(self):
        nlp = spacy.blank("clinlp")
        nlp("dit is een test")
