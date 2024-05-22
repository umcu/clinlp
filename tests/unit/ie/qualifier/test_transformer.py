import statistics

import pytest
import spacy

import clinlp  # noqa
from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier import (
    ExperiencerTransformer,
    NegationTransformer,
    QualifierTransformer,
)
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR


@pytest.fixture
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")
    ruler = nlp.add_pipe("span_ruler", config={"spans_key": SPANS_KEY})
    ruler.add_patterns([{"label": "symptoom", "pattern": "SYMPTOOM"}])
    return nlp


@pytest.fixture
def text():
    return "De patient had geen SYMPTOOM, ondanks dat zij dit eerder wel had."


class TestQualifierTransformer:
    def test_get_ent_window(self, text, nlp):
        # Arrange
        doc = nlp(text)
        span = doc.spans[SPANS_KEY][0]

        # Act & Assert
        assert QualifierTransformer._get_ent_window(span, token_window=1) == (
            "geen SYMPTOOM,",
            5,
            13,
        )
        assert QualifierTransformer._get_ent_window(span, token_window=2) == (
            "had geen SYMPTOOM, ondanks",
            9,
            17,
        )
        assert QualifierTransformer._get_ent_window(span, token_window=32) == (
            "De patient had geen SYMPTOOM, ondanks dat zij dit eerder wel had.",
            20,
            28,
        )

    def test_trim_ent_boundaries(self):
        # Arrange, Act & Assert
        assert QualifierTransformer._trim_ent_boundaries("geen SYMPTOOM,", 5, 13) == (
            "geen SYMPTOOM,",
            5,
            13,
        )
        assert QualifierTransformer._trim_ent_boundaries("geen SYMPTOOM,", 4, 13) == (
            "geen SYMPTOOM,",
            5,
            13,
        )
        assert QualifierTransformer._trim_ent_boundaries(
            "had geen SYMPTOOM, ondanks", 8, 17
        ) == (
            "had geen SYMPTOOM, ondanks",
            9,
            17,
        )
        assert QualifierTransformer._trim_ent_boundaries(
            "had geen SYMPTOOM, ondanks", 8, 19
        ) == (
            "had geen SYMPTOOM, ondanks",
            9,
            18,
        )

    def test_fill_ent_placeholder(self):
        # Arrange, Act & Assert
        assert QualifierTransformer._fill_ent_placeholder(
            "geen SYMPTOOM,", 5, 13, placeholder="SYMPTOOM"
        ) == ("geen SYMPTOOM,", 5, 13)
        assert QualifierTransformer._fill_ent_placeholder(
            "geen SYMPTOOM,", 5, 13, placeholder="X"
        ) == (
            "geen X,",
            5,
            6,
        )

    def test_prepare_ent(self, nlp, text):
        # Arrange
        doc = nlp(text)
        QualifierTransformer.__abstractmethods__ = set()
        qt = QualifierTransformer(token_window=3, placeholder="X")

        # Act
        text, ent_start_char, ent_end_char = qt._prepare_ent(doc.spans[SPANS_KEY][0])

        # Assert
        assert text == "patient had geen X, ondanks dat"
        assert ent_start_char == 17
        assert ent_end_char == 18


class TestNegationTransformer:
    def test_predict(self, nlp):
        # Arrange
        n = NegationTransformer(nlp=nlp)

        # Act & Assert
        assert (
            n._predict(
                text="geen hoesten,",
                ent_start_char=5,
                ent_end_char=11,
                prob_indices=[0, 2],
                prob_aggregator=statistics.mean,
            )
            > 0.9
        )
        assert (
            n._predict(
                text="wel hoesten,",
                ent_start_char=4,
                ent_end_char=10,
                prob_indices=[0, 2],
                prob_aggregator=statistics.mean,
            )
            < 0.1
        )

    def test_detect_qualifiers_1(self, nlp):
        # Arrange
        n = NegationTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")

        # Act
        n(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Presence.Absent"
        }

    def test_detect_qualifiers_small_window(self, nlp):
        # Arrange
        n = NegationTransformer(nlp=nlp, token_window=1, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")

        # Act
        n(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Presence.Present"
        }

    def test_detect_qualifiers_without_negation(self, nlp):
        # Arrange
        n = NegationTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De patient had juist wel last van SYMPTOOM.")

        # Act
        n(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Presence.Present"
        }


class TestExperiencerTransformer:
    def test_predict(self, nlp):
        # Arrange
        n = ExperiencerTransformer(nlp=nlp)

        # Act & Assert
        assert (
            n._predict(
                text="broer heeft aandoening,",
                ent_start_char=12,
                ent_end_char=22,
                prob_indices=[1, 3],
                prob_aggregator=statistics.mean,
            )
            > 0.9
        )
        assert (
            n._predict(
                text="patient heeft aandoening,",
                ent_start_char=14,
                ent_end_char=24,
                prob_indices=[1, 3],
                prob_aggregator=statistics.mean,
            )
            < 0.1
        )

    def test_detect_qualifiers_1(self, nlp):
        # Arrange
        n = ExperiencerTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")

        # Act
        n(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Experiencer.Patient"
        }

    def test_detect_qualifiers_small_window(self, nlp):
        # Arrange
        n = ExperiencerTransformer(nlp=nlp, token_window=1, placeholder="X")
        doc = nlp("De patient had geen last van SYMPTOOM.")

        # Act
        n(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Experiencer.Patient"
        }

    def test_detect_qualifiers_referring_to_family(self, nlp):
        # Arrange
        n = ExperiencerTransformer(nlp=nlp, token_window=32, placeholder="X")
        doc = nlp("De broer van de patient had last van SYMPTOOM.")

        # Act
        n(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Experiencer.Family"
        }
