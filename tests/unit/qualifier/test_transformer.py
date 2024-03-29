import statistics

import pytest
import spacy

import clinlp  # noqa
from clinlp.qualifier import NegationTransformer, ExperiencerTransformer
from clinlp.qualifier.qualifier import ATTR_QUALIFIERS_STR


@pytest.fixture
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "symptoom", "pattern": "SYMPTOOM"}])
    return nlp


@pytest.fixture
def text():
    return "De patient had geen SYMPTOOM, ondanks dat zij dit eerder wel had."


class TestNegationTransformer:
    def test_get_ent_window(self, nlp, text):
        doc = nlp(text)
        span = doc.ents[0]
        n = NegationTransformer(nlp=nlp)

        assert n._get_ent_window(span, token_window=1) == ("geen SYMPTOOM,", 5, 13)
        assert n._get_ent_window(span, token_window=2) == (
            "had geen SYMPTOOM, ondanks",
            9,
            17,
        )
        assert n._get_ent_window(span, token_window=32) == (
            "De patient had geen SYMPTOOM, ondanks dat zij dit eerder wel had.",
            20,
            28,
        )

    def test_trim_ent_boundaries(self, nlp):
        n = NegationTransformer(nlp=nlp)

        assert n._trim_ent_boundaries("geen SYMPTOOM,", 5, 13) == (
            "geen SYMPTOOM,",
            5,
            13,
        )
        assert n._trim_ent_boundaries("geen SYMPTOOM,", 4, 13) == (
            "geen SYMPTOOM,",
            5,
            13,
        )
        assert n._trim_ent_boundaries("had geen SYMPTOOM, ondanks", 8, 17) == (
            "had geen SYMPTOOM, ondanks",
            9,
            17,
        )
        assert n._trim_ent_boundaries("had geen SYMPTOOM, ondanks", 8, 19) == (
            "had geen SYMPTOOM, ondanks",
            9,
            18,
        )

    def test_fill_ent_placeholder(self, nlp):
        n = NegationTransformer(nlp=nlp)

        assert n._fill_ent_placeholder(
            "geen SYMPTOOM,", 5, 13, placeholder="SYMPTOOM"
        ) == ("geen SYMPTOOM,", 5, 13)
        assert n._fill_ent_placeholder("geen SYMPTOOM,", 5, 13, placeholder="X") == (
            "geen X,",
            5,
            6,
        )

    def test_get_negation_prob(self, nlp):
        n = NegationTransformer(nlp=nlp)

        assert (
            n._get_negation_prob(
                text="geen hoesten,",
                ent_start_char=5,
                ent_end_char=11,
                probas_aggregator=statistics.mean,
            )
            > 0.9
        )
        assert (
            n._get_negation_prob(
                text="wel hoesten,",
                ent_start_char=4,
                ent_end_char=10,
                probas_aggregator=statistics.mean,
            )
            < 0.1
        )

    def test_detect_qualifiers_1(self, nlp):
        n = NegationTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")
        n(doc)

        assert len(doc.ents) == 1
        assert getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR) == {"Negation.Negated"}

    def test_detect_qualifiers_small_window(self, nlp):
        n = NegationTransformer(nlp=nlp, token_window=1, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")
        n(doc)

        assert len(doc.ents) == 1
        assert getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR) == {"Negation.Affirmed"}

    def test_detect_qualifiers_without_negation(self, nlp):
        n = NegationTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De patient had juist wel last van SYMPTOOM.")
        n(doc)

        assert len(doc.ents) == 1
        assert getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR) == {"Negation.Affirmed"}


class TestExperiencerTransformer:
    def test_get_ent_window(self, nlp, text):
        doc = nlp(text)
        span = doc.ents[0]
        n = ExperiencerTransformer(nlp=nlp)

        assert n._get_ent_window(span, token_window=1) == ("geen SYMPTOOM,", 5, 13)
        assert n._get_ent_window(span, token_window=2) == (
            "had geen SYMPTOOM, ondanks",
            9,
            17,
        )
        assert n._get_ent_window(span, token_window=32) == (
            "De patient had geen SYMPTOOM, ondanks dat zij dit eerder wel had.",
            20,
            28,
        )

    def test_trim_ent_boundaries(self, nlp):
        n = ExperiencerTransformer(nlp=nlp)

        assert n._trim_ent_boundaries("geen SYMPTOOM,", 5, 13) == (
            "geen SYMPTOOM,",
            5,
            13,
        )
        assert n._trim_ent_boundaries("geen SYMPTOOM,", 4, 13) == (
            "geen SYMPTOOM,",
            5,
            13,
        )
        assert n._trim_ent_boundaries("had geen SYMPTOOM, ondanks", 8, 17) == (
            "had geen SYMPTOOM, ondanks",
            9,
            17,
        )
        assert n._trim_ent_boundaries("had geen SYMPTOOM, ondanks", 8, 19) == (
            "had geen SYMPTOOM, ondanks",
            9,
            18,
        )

    def test_fill_ent_placeholder(self, nlp):
        n = ExperiencerTransformer(nlp=nlp)

        assert n._fill_ent_placeholder(
            "geen SYMPTOOM,", 5, 13, placeholder="SYMPTOOM"
        ) == ("geen SYMPTOOM,", 5, 13)
        assert n._fill_ent_placeholder("geen SYMPTOOM,", 5, 13, placeholder="X") == (
            "geen X,",
            5,
            6,
        )

    def test_get_negation_prob(self, nlp):
        n = ExperiencerTransformer(nlp=nlp)

        assert (
            n._get_experiencer_prob(
                text="familiaire aandoening,",
                ent_start_char=11,
                ent_end_char=21,
                probas_aggregator=statistics.mean,
            )
            < 0.1
        )
        assert (
            n._get_experiencer_prob(
                text="patient heeft aandoening,",
                ent_start_char=14,
                ent_end_char=24,
                probas_aggregator=statistics.mean,
            )
            > 0.9
        )

    def test_detect_qualifiers_1(self, nlp):
        n = ExperiencerTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")
        n(doc)

        assert len(doc.ents) == 1
        assert getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR) == {"Experiencer.Patient"}

    def test_detect_qualifiers_small_window(self, nlp):
        n = ExperiencerTransformer(nlp=nlp, token_window=1, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")
        n(doc)

        assert len(doc.ents) == 1
        assert getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR) == {"Experiencer.Patient"}

    def test_detect_qualifiers_referring_to_other(self, nlp):
        n = ExperiencerTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De broer van de patient had last van SYMPTOOM.")
        n(doc)

        assert len(doc.ents) == 1
        assert getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR) == {"Experiencer.Other"}