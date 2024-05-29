import statistics

import pytest

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier import (
    ExperiencerTransformer,
    NegationTransformer,
    QualifierTransformer,
)
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR


class TestQualifierTransformer:
    @pytest.mark.parametrize(
        ("token_window", "expected_text", "expected_start", "expected_end"),
        [
            (1, "geen ENTITY,", 5, 11),
            (2, "had geen ENTITY, ondanks", 9, 15),
            (
                32,
                "De patient had geen ENTITY, ondanks dat zij dit eerder wel had.",
                20,
                26,
            ),
        ],
    )
    def test_get_ent_window(
        self,
        nlp_entity,
        token_window,
        expected_text,
        expected_start,
        expected_end,
    ):
        # Arrange
        text = "De patient had geen ENTITY, ondanks dat zij dit eerder wel had."
        doc = nlp_entity(text)
        span = doc.spans[SPANS_KEY][0]

        # Act
        text, start, end = QualifierTransformer._get_ent_window(
            span, token_window=token_window
        )

        # Assert
        assert text == expected_text
        assert start == expected_start
        assert end == expected_end

    @pytest.mark.parametrize(
        ("text", "start", "end", "expected_text", "expected_start", "expected_end"),
        [
            ("geen SYMPTOOM,", 5, 13, "geen SYMPTOOM,", 5, 13),
            ("geen SYMPTOOM,", 4, 13, "geen SYMPTOOM,", 5, 13),
            ("had geen SYMPTOOM, ondanks", 8, 17, "had geen SYMPTOOM, ondanks", 9, 17),
            ("had geen SYMPTOOM, ondanks", 8, 19, "had geen SYMPTOOM, ondanks", 9, 18),
        ],
    )
    def test_trim_ent_boundaries(
        self, text, start, end, expected_text, expected_start, expected_end
    ):
        # Act
        text, start, end = QualifierTransformer._trim_ent_boundaries(text, start, end)

        # Assert
        assert text == expected_text
        assert start == expected_start
        assert end == expected_end

    @pytest.mark.parametrize(
        (
            "text",
            "start",
            "end",
            "placeholder",
            "expected_text",
            "expected_start",
            "expected_end",
        ),
        [
            ("geen SYMPTOOM,", 5, 13, "SYMPTOOM", "geen SYMPTOOM,", 5, 13),
            ("geen SYMPTOOM,", 5, 13, "X", "geen X,", 5, 6),
        ],
    )
    def test_fill_ent_placeholder(
        self, text, start, end, placeholder, expected_text, expected_start, expected_end
    ):
        # Act
        text, start, end = QualifierTransformer._fill_ent_placeholder(
            text, start, end, placeholder=placeholder
        )

        # Assert
        assert text == expected_text
        assert start == expected_start
        assert end == expected_end

    def test_prepare_ent(self, nlp_entity):
        # Arrange
        text = "De patient had geen ENTITY, ondanks dat zij dit eerder wel had."
        doc = nlp_entity(text)
        QualifierTransformer.__abstractmethods__ = set()
        qt = QualifierTransformer(token_window=3, placeholder="X")

        # Act
        text, ent_start_char, ent_end_char = qt._prepare_ent(doc.spans[SPANS_KEY][0])

        # Assert
        assert text == "patient had geen X, ondanks dat"
        assert ent_start_char == 17
        assert ent_end_char == 18


class TestNegationTransformer:
    def test_predict_absent(self, nlp_entity):
        # Arrange
        nt = NegationTransformer(nlp=nlp_entity)

        # Act
        prediction = nt._predict(
            text="geen hoesten,",
            ent_start_char=5,
            ent_end_char=11,
            prob_indices=[0, 2],
            prob_aggregator=statistics.mean,
        )

        # Assert
        assert prediction > 0.9

    def test_predict_present(self, nlp_entity):
        # Arrange
        nt = NegationTransformer(nlp=nlp_entity)

        # Act
        prediction = nt._predict(
            text="wel hoesten,",
            ent_start_char=4,
            ent_end_char=10,
            prob_indices=[0, 2],
            prob_aggregator=statistics.mean,
        )

        # Assert
        assert prediction < 0.1

    def test_detect_qualifiers(self, nlp_entity):
        # Arrange
        nt = NegationTransformer(nlp=nlp_entity, token_window=32, placeholder="X")
        doc = nlp_entity("De patient had geen last van ENTITY.")

        # Act
        nt(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Presence.Absent"
        }

    def test_detect_qualifiers_small_window(self, nlp_entity):
        # Arrange
        nt = NegationTransformer(nlp=nlp_entity, token_window=1, placeholder="X")
        doc = nlp_entity("De patient had geen last van ENTITY.")

        # Act
        nt(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Presence.Present"
        }

    def test_detect_qualifiers_present(self, nlp_entity):
        # Arrange
        nt = NegationTransformer(nlp=nlp_entity, token_window=32, placeholder="X")
        doc = nlp_entity("De patient had juist wel last van ENTITY.")

        # Act
        nt(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Presence.Present"
        }


class TestExperiencerTransformer:
    def test_predict_family(self, nlp_entity):
        # Arrange
        et = ExperiencerTransformer(nlp=nlp_entity)

        # Act
        prediction = et._predict(
            text="broer heeft aandoening,",
            ent_start_char=12,
            ent_end_char=22,
            prob_indices=[1, 3],
            prob_aggregator=statistics.mean,
        )

        # Assert
        assert prediction > 0.9

    def test_predict_patient(self, nlp_entity):
        # Arrange
        et = ExperiencerTransformer(nlp=nlp_entity)

        # Act
        prediction = et._predict(
            text="patient heeft aandoening,",
            ent_start_char=14,
            ent_end_char=24,
            prob_indices=[1, 3],
            prob_aggregator=statistics.mean,
        )

        # Assert
        assert prediction < 0.1

    def test_detect_qualifiers(self, nlp_entity):
        # Arrange
        et = ExperiencerTransformer(nlp=nlp_entity, token_window=32, placeholder="X")
        doc = nlp_entity("De patient had geen last van ENTITY.")

        # Act
        et(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Experiencer.Patient"
        }

    def test_detect_qualifiers_small_window(self, nlp_entity):
        # Arrange
        et = ExperiencerTransformer(nlp=nlp_entity, token_window=1, placeholder="X")
        doc = nlp_entity("De patient had geen last van ENTITY.")

        # Act
        et(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Experiencer.Patient"
        }

    def test_detect_qualifiers_family(self, nlp_entity):
        # Arrange
        et = ExperiencerTransformer(nlp=nlp_entity, token_window=32, placeholder="X")
        doc = nlp_entity("De broer van de patient had last van ENTITY.")

        # Act
        et(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR) == {
            "Experiencer.Family"
        }
