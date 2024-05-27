import json
import pickle

import pytest

import clinlp  # noqa: F401
from clinlp.metrics.ie import (
    Annotation,
    Document,
    InfoExtractionDataset,
    InfoExtractionMetrics,
)


@pytest.fixture
def mctrainer_data():
    with open("tests/data/medcattrainer_export.json", "rb") as f:
        return json.load(f)


@pytest.fixture
def mctrainer_dataset(mctrainer_data):
    return InfoExtractionDataset.from_medcattrainer(data=mctrainer_data)


@pytest.fixture
def clinlp_docs():
    with open("tests/data/clinlp_docs.pickle", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def clinlp_dataset(clinlp_docs):
    ids = list(f"doc_{x}" for x in range(0, 15))

    return InfoExtractionDataset.from_clinlp_docs(nlp_docs=clinlp_docs, ids=ids)


class TestAnnotation:
    def test_annotation_nervaluate(self):
        # Arrange
        ann = Annotation(text="test", start=0, end=5, label="test")

        # Act
        nervaluate = ann.to_nervaluate()

        # Assert
        assert nervaluate == {
            "text": "test",
            "start": 0,
            "end": 5,
            "label": "test",
        }

    def test_annotation_lstrip(self):
        # Arrange
        ann = Annotation(text=" test", start=0, end=5, label="test")

        # Act
        ann.lstrip()

        # Assert
        assert ann == Annotation(text="test", start=1, end=5, label="test")

    def test_annotation_rstrip(self):
        # Arrange
        ann = Annotation(text="test,", start=0, end=5, label="test")

        # Act
        ann.rstrip()

        # Assert
        assert ann == Annotation(text="test", start=0, end=4, label="test")

    def test_annotation_strip(self):
        # Arrange
        ann = Annotation(text=" test,", start=0, end=6, label="test")

        # Act
        ann.strip()

        # Assert
        assert ann == Annotation(text="test", start=1, end=5, label="test")

    def test_annotation_qualifier_names(self):
        # Arrange
        ann = Annotation(
            text="test",
            start=0,
            end=4,
            label="test",
            qualifiers=[
                {"name": "Negation", "value": "Affirmed"},
                {"name": "Experiencer", "value": "Other"},
            ],
        )

        # Act
        qualifier_names = ann.qualifier_names

        # Assert
        assert qualifier_names == {"Negation", "Experiencer"}

    def test_annotation_get_qualifier_by_name(self):
        # Arrange
        ann = Annotation(
            text="test",
            start=0,
            end=4,
            label="test",
            qualifiers=[
                {"name": "Negation", "value": "Affirmed"},
                {"name": "Experiencer", "value": "Other"},
            ],
        )

        # Act
        qualifier = ann.get_qualifier_by_name(qualifier_name="Experiencer")

        # Assert
        assert qualifier == {"name": "Experiencer", "value": "Other"}


class TestDocument:
    def test_document_nervaluate(self):
        # Arrange
        doc = Document(
            identifier="1",
            text="test1 and test2",
            annotations=[
                Annotation(text="test1", start=0, end=5, label="test1"),
                Annotation(text="test2", start=10, end=15, label="test2"),
            ],
        )

        # Act
        nervaluate = doc.to_nervaluate()

        # Assert
        assert nervaluate == [
            {"text": "test1", "start": 0, "end": 5, "label": "test1"},
            {"text": "test2", "start": 10, "end": 15, "label": "test2"},
        ]

    def test_document_labels(self):
        # Arrange
        doc = Document(
            identifier="1",
            text="test1 and test2",
            annotations=[
                Annotation(text="test1", start=0, end=5, label="test1"),
                Annotation(text="test2", start=10, end=15, label="test2"),
            ],
        )

        # Act
        labels = doc.labels()

        # Assert
        assert labels == {"test1", "test2"}

    def test_document_labels_with_filter(self):
        # Arrange
        doc = Document(
            identifier="1",
            text="test1 and test2",
            annotations=[
                Annotation(text="test1", start=0, end=5, label="test1"),
                Annotation(text="test2", start=10, end=15, label="test2"),
            ],
        )

        # Act
        labels = doc.labels(ann_filter=(lambda ann: ann.start > 5))

        # Assert
        assert labels == {"test2"}


class TestDataset:
    def test_dataset_from_medcattrainer_docs(self, mctrainer_data):
        # Act
        ied = InfoExtractionDataset.from_medcattrainer(data=mctrainer_data)

        # Assert
        assert len(ied.docs) == 14
        assert ied.docs[0].text == "patient had geen anemie"
        assert len(ied.docs[0].annotations) == 1
        assert ied.docs[3].text == "patient had een prematuur adempatroon"
        assert len(ied.docs[3].annotations) == 0
        assert ied.docs[6].text == "na fototherapie verminderde hyperbillirubinaemie"
        assert len(ied.docs[6].annotations) == 2

    def test_dataset_from_medcattrainer_annotations(self, mctrainer_data):
        # Act
        ied = InfoExtractionDataset.from_medcattrainer(data=mctrainer_data)

        # Assert
        assert ied.docs[0].annotations[0].text == "anemie"
        assert ied.docs[0].annotations[0].start == 17
        assert ied.docs[0].annotations[0].end == 23
        assert ied.docs[0].annotations[0].label == "C0002871_anemie"
        assert ied.docs[6].annotations[1].text == "hyperbillirubinaemie"
        assert ied.docs[6].annotations[1].start == 28
        assert ied.docs[6].annotations[1].end == 48
        assert ied.docs[6].annotations[1].label == "C0020433_hyperbilirubinemie"

    def test_dataset_from_medcatrainer_qualifiers(self, mctrainer_data):
        # Arrange
        ied = InfoExtractionDataset.from_medcattrainer(data=mctrainer_data)

        # Act
        qualifiers = ied.docs[0].annotations[0].qualifiers

        # Assert
        assert qualifiers == [
            {"name": "Temporality", "value": "Current", "is_default": True},
            {"name": "Plausibility", "value": "Plausible", "is_default": True},
            {"name": "Experiencer", "value": "Patient", "is_default": True},
            {"name": "Negation", "value": "Negated", "is_default": False},
        ]

    def test_dataset_from_clinlp_docs(self, clinlp_docs):
        # Act
        ied = InfoExtractionDataset.from_clinlp_docs(nlp_docs=clinlp_docs)

        # Assert
        assert len(ied.docs) == 14
        assert ied.docs[0].text == "patient had geen anemie"
        assert len(ied.docs[0].annotations) == 1
        assert ied.docs[3].text == "patient had een prematuur adempatroon"
        assert len(ied.docs[3].annotations) == 1
        assert ied.docs[6].text == "na fototherapie verminderde hyperbillirubinaemie"
        assert len(ied.docs[6].annotations) == 2

    def test_dataset_from_clinlp_annotations(self, clinlp_docs):
        # Act
        ied = InfoExtractionDataset.from_clinlp_docs(nlp_docs=clinlp_docs)

        # Assert
        assert ied.docs[0].annotations[0].text == "anemie"
        assert ied.docs[0].annotations[0].start == 17
        assert ied.docs[0].annotations[0].end == 23
        assert ied.docs[0].annotations[0].label == "C0002871_anemie"
        assert ied.docs[6].annotations[1].text == "hyperbillirubinaemie"
        assert ied.docs[6].annotations[1].start == 28
        assert ied.docs[6].annotations[1].end == 48
        assert ied.docs[6].annotations[1].label == "C0020433_hyperbilirubinemie"

    def test_dataset_from_clinlp_qualifiers(self, clinlp_docs):
        # Arrange
        ied = InfoExtractionDataset.from_clinlp_docs(nlp_docs=clinlp_docs)

        # Act
        qualifiers = sorted(
            ied.docs[0].annotations[0].qualifiers, key=lambda q: q["name"]
        )

        # Assert
        assert qualifiers == [
            {"name": "Experiencer", "value": "Patient", "is_default": True},
            {"name": "Negation", "value": "Negated", "is_default": False},
            {"name": "Plausibility", "value": "Plausible", "is_default": True},
            {"name": "Temporality", "value": "Current", "is_default": True},
        ]

    def test_dataset_nervaluate(self):
        # Arrange
        ied = InfoExtractionDataset(
            docs=[
                Document(
                    identifier="1",
                    text="test1",
                    annotations=[
                        Annotation(
                            text="test1",
                            start=0,
                            end=5,
                            label="test1",
                            qualifiers=[
                                {
                                    "name": "Negation",
                                    "value": "Negated",
                                    "is_default": False,
                                }
                            ],
                        ),
                    ],
                ),
                Document(
                    identifier="2",
                    text="test2",
                    annotations=[
                        Annotation(text="test2", start=0, end=5, label="test2"),
                    ],
                ),
            ]
        )

        # Act
        nervaluate = ied.to_nervaluate()

        # Assert
        assert nervaluate == [
            [{"text": "test1", "start": 0, "end": 5, "label": "test1"}],
            [{"text": "test2", "start": 0, "end": 5, "label": "test2"}],
        ]

    def test_dataset_to_nervaluate_with_filter(self, mctrainer_dataset):
        # Arrange
        def ann_filter(ann):
            return any(not qualifier["is_default"] for qualifier in ann.qualifiers)

        # Act
        to_nervaluate = mctrainer_dataset.to_nervaluate(ann_filter=ann_filter)

        # Assert
        assert to_nervaluate[0] == [
            {"end": 23, "label": "C0002871_anemie", "start": 17, "text": "anemie"}
        ]
        assert to_nervaluate[1] == []

    def test_infer_default_qualifiers(self, mctrainer_dataset):
        # Act
        default_qualifiers = mctrainer_dataset.infer_default_qualifiers()

        # Assert
        assert default_qualifiers == {
            "Negation": "Affirmed",
            "Experiencer": "Patient",
            "Temporality": "Current",
            "Plausibility": "Plausible",
        }

    def test_num_docs(self, mctrainer_dataset):
        # Act
        num_docs = mctrainer_dataset.num_docs()

        # Assert
        assert num_docs == 14

    def test_num_annotations(self, mctrainer_dataset):
        # Act
        num_annotations = mctrainer_dataset.num_annotations()

        # Assert
        assert num_annotations == 13

    def test_span_counts(self, mctrainer_dataset):
        # Act
        num_span_counts = len(mctrainer_dataset.span_counts())

        # Assert
        assert num_span_counts == 11

    def test_span_counts_n_spans(self, mctrainer_dataset):
        # Act
        span_counts = mctrainer_dataset.span_counts(n_spans=3)

        # Assert
        assert span_counts == {
            "anemie": 2,
            "bloeding": 2,
            "prematuriteit": 1,
        }

    def test_span_counts_callback(self, mctrainer_dataset):
        # Act
        span_counts = mctrainer_dataset.span_counts(
            n_spans=3, span_callback=lambda x: x.upper()
        )

        # Assert
        assert span_counts == {
            "ANEMIE": 2,
            "BLOEDING": 2,
            "PREMATURITEIT": 1,
        }

    def test_label_counts(self, mctrainer_dataset):
        # Act
        num_label_counts = len(mctrainer_dataset.label_counts())

        # Assert
        assert num_label_counts == 9

    def test_label_counts_n_labels(self, mctrainer_dataset):
        # Act
        label_counts = mctrainer_dataset.label_counts(n_labels=3)

        # Assert
        assert label_counts == {
            "C0002871_anemie": 2,
            "C0151526_prematuriteit": 2,
            "C0270191_intraventriculaire_bloeding": 2,
        }

    def test_label_counts_callback(self, mctrainer_dataset):
        # Act
        label_counts = mctrainer_dataset.label_counts(
            n_labels=3, label_callback=lambda x: x[x.index("_") + 1 :]
        )

        # Assert
        assert label_counts == {
            "anemie": 2,
            "prematuriteit": 2,
            "intraventriculaire_bloeding": 2,
        }

    def test_qualifier_counts(self, mctrainer_dataset):
        # Act
        qualifier_counts = mctrainer_dataset.qualifier_counts()

        # Assert
        assert qualifier_counts == {
            "Experiencer": {"Patient": 12, "Other": 1},
            "Negation": {"Affirmed": 11, "Negated": 2},
            "Plausibility": {"Plausible": 11, "Hypothetical": 2},
            "Temporality": {"Current": 11, "Historical": 2},
        }

    def test_stats(self, mctrainer_dataset):
        # Act
        stats = mctrainer_dataset.stats()

        # Assert
        assert stats["num_docs"] == mctrainer_dataset.num_docs()
        assert stats["num_annotations"] == mctrainer_dataset.num_annotations()
        assert stats["span_counts"] == mctrainer_dataset.span_counts()
        assert stats["label_counts"] == mctrainer_dataset.label_counts()
        assert stats["qualifier_counts"] == mctrainer_dataset.qualifier_counts()

    def test_stats_with_kwargs(self, mctrainer_dataset):
        # Arrange
        n_labels = 1
        span_callback = lambda x: x.upper()  # noqa: E731

        # Act
        stats = mctrainer_dataset.stats(
            n_labels=n_labels, span_callback=span_callback, unused_argument=None
        )

        # Assert
        assert stats["num_docs"] == mctrainer_dataset.num_docs()
        assert stats["num_annotations"] == mctrainer_dataset.num_annotations()
        assert stats["span_counts"] == mctrainer_dataset.span_counts(
            span_callback=span_callback
        )
        assert stats["label_counts"] == mctrainer_dataset.label_counts(
            n_labels=n_labels
        )
        assert stats["qualifier_counts"] == mctrainer_dataset.qualifier_counts()


class TestMetrics:
    def test_entity_metrics(self, mctrainer_dataset, clinlp_dataset):
        # Arrange
        iem = InfoExtractionMetrics(mctrainer_dataset, clinlp_dataset)

        # Act
        metrics = iem.entity_metrics()

        # Assert
        assert list(metrics.keys()) == ["ent_type", "partial", "strict", "exact"]
        assert metrics["strict"]["actual"] == 13
        assert metrics["strict"]["correct"] == 10
        assert metrics["strict"]["precision"] == 0.7692307692307693
        assert metrics["strict"]["recall"] == 0.7692307692307693
        assert metrics["strict"]["f1"] == 0.7692307692307693

    def test_entity_metrics_filter(self, mctrainer_dataset, clinlp_dataset):
        # Arrange
        iem = InfoExtractionMetrics(mctrainer_dataset, clinlp_dataset)

        def filter_default(ann):
            return all(qualifier["is_default"] for qualifier in ann.qualifiers)

        # Act
        metrics = iem.entity_metrics(ann_filter=filter_default)

        # Assert
        assert metrics["strict"]["actual"] == 6
        assert metrics["strict"]["correct"] == 4
        assert metrics["strict"]["precision"] == 0.6666666666666666
        assert metrics["strict"]["recall"] == 0.5
        assert metrics["strict"]["f1"] == 0.5714285714285715

    def test_entity_metrics_classes(self, mctrainer_dataset, clinlp_dataset):
        # Arrange
        iem = InfoExtractionMetrics(mctrainer_dataset, clinlp_dataset)

        # Act
        metrics = iem.entity_metrics(classes=True)

        # Assert
        assert len(metrics) == 9
        assert metrics["C0151526_prematuriteit"]["strict"]["actual"] == 2
        assert metrics["C0151526_prematuriteit"]["strict"]["correct"] == 1
        assert metrics["C0151526_prematuriteit"]["strict"]["precision"] == 0.5
        assert metrics["C0151526_prematuriteit"]["strict"]["recall"] == 0.5
        assert metrics["C0151526_prematuriteit"]["strict"]["f1"] == 0.5

    def test_qualifier_metrics(self, mctrainer_dataset, clinlp_dataset):
        # Arrange
        iem = InfoExtractionMetrics(mctrainer_dataset, clinlp_dataset)

        # Act
        metrics = iem.qualifier_metrics()

        # Assert
        assert metrics["Negation"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 2,
            "n_pos_true": 2,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        assert metrics["Experiencer"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 1,
            "n_pos_true": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        assert metrics["Plausibility"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 3,
            "n_pos_true": 2,
            "precision": 0.6666666666666666,
            "recall": 1.0,
            "f1": 0.8,
        }
        assert metrics["Temporality"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 1,
            "n_pos_true": 2,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 0.6666666666666666,
        }

    def test_qualifier_misses(self, mctrainer_dataset, clinlp_dataset):
        # Arrange
        iem = InfoExtractionMetrics(mctrainer_dataset, clinlp_dataset)

        # Act
        metrics = iem.qualifier_metrics()

        # Assert
        assert len(metrics["Negation"]["misses"]) == 0
        assert len(metrics["Experiencer"]["misses"]) == 0
        assert len(metrics["Plausibility"]["misses"]) == 1
        assert len(metrics["Temporality"]["misses"]) == 1

    def test_create_metrics_unequal_length(self, mctrainer_dataset, clinlp_dataset):
        # Arrange
        mctrainer_dataset.docs = mctrainer_dataset.docs[:-2]

        # Assert
        with pytest.raises(ValueError):
            # Act
            _ = InfoExtractionMetrics(mctrainer_dataset, clinlp_dataset)

    def test_create_metrics_unequal_names(self, mctrainer_dataset, clinlp_dataset):
        # Arrange
        mctrainer_dataset.docs[0].identifier = "test"

        # Assert
        with pytest.raises(ValueError):
            # Act
            _ = InfoExtractionMetrics(mctrainer_dataset, clinlp_dataset)
