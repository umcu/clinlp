"""Classes and functions for evaluating information extraction tasks."""

import inspect
import itertools
import json
import pathlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar

import nervaluate
from sklearn.metrics import f1_score, precision_score, recall_score
from spacy.language import Doc

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier import Qualifier


@dataclass
class Annotation:
    """An annotation in a document."""

    text: str
    """The text/str span of this annotation."""

    start: int
    """The start char."""

    end: int
    """The end char."""

    label: str
    """The label/tag."""

    qualifiers: list[Qualifier] | None = None
    """The applicable qualifiers."""

    def lstrip(self, chars: str = " ,") -> None:
        """
        Strip punctuation and whitespaces from the beginning of the annotation.

        Parameters
        ----------
        chars
            The characters to strip from the beginning.
        """
        self.start += len(self.text) - len(self.text.lstrip(chars))
        self.text = self.text.lstrip(chars)

    def rstrip(self, chars: str = " ,") -> None:
        """
        Strip punctuation and whitespaces from the end of the annotation.

        Parameters
        ----------
        chars
            The characters to strip from the end.
        """
        self.end -= len(self.text) - len(self.text.rstrip(chars))
        self.text = self.text.rstrip(chars)

    def strip(self, chars: str = " ,") -> None:
        """
        Strip punctuation and whitespaces from the beginning and end of the annotation.

        Parameters
        ----------
        chars
            The characters to strip from the beginning and end.
        """
        self.lstrip(chars=chars)
        self.rstrip(chars=chars)

    def to_nervaluate(self) -> dict:
        """
        Convert to ``nervaluate`` format.

        Returns
        -------
        ``dict``
            A dictionary with the items ``nervaluate`` expects.
        """
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "label": self.label,
        }

    def to_dict(self) -> dict:
        """
        Convert to dictionary format.

        Returns
        -------
        ``dict``
            A dictionary with the items of this annotation.
        """
        output = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "label": self.label,
        }

        if self.qualifiers is not None:
            output["qualifiers"] = [
                {"name": q.name, "value": q.value} for q in self.qualifiers
            ]

        return output

    @property
    def qualifier_names(self) -> set[str]:
        """
        Obtain the unique qualifier names for this annotation.

        Returns
        -------
        ``set[str]``
            A set of unique qualifier names, e.g. {"Presence", "Experiencer"}.
        """
        if self.qualifiers is None:
            return {}

        return {qualifier.name for qualifier in self.qualifiers}

    def get_qualifier_by_name(self, qualifier_name: str) -> Qualifier:
        """
        Get a qualifier by name.

        Parameters
        ----------
        qualifier_name
            The name of the qualifier.

        Returns
        -------
        ``Qualifier``
            The qualifier with the provided name.

        Raises
        ------
        KeyError
            If no qualifier with the provided name exists.
        """
        for qualifier in self.qualifiers:
            if qualifier.name == qualifier_name:
                return qualifier

        msg = f"No qualifier with name {qualifier_name}."
        raise KeyError(msg)


@dataclass
class Document:
    """Document (any text) with annotations."""

    identifier: str
    """Any identifier for the document."""

    text: str
    """The text."""

    annotations: list[Annotation]
    """A list of annotations."""

    def to_nervaluate(
        self, ann_filter: Callable[[Annotation], bool] | None = None
    ) -> list[dict]:
        """
        Convert to ``nervaluate`` format.

        Parameters
        ----------
        ann_filter
            A filter to apply to annotations. Should map the annotations to ``True``
            if they should be included, ``False`` otherwise.

        Returns
        -------
        ``list[dict]``
            A list of dictionaries corresponding to annotations.
        """
        ann_filter = ann_filter or (lambda _: True)

        return [ann.to_nervaluate() for ann in self.annotations if ann_filter(ann)]

    def to_dict(self) -> dict:
        """
        Convert to dictionary format.

        Returns
        -------
        ``dict``
            A dictionary with the items of this document.
        """
        return {
            "identifier": self.identifier,
            "text": self.text,
            "annotations": [ann.to_dict() for ann in self.annotations],
        }

    def labels(
        self, ann_filter: Callable[[Annotation], bool] | None = None
    ) -> set[str]:
        """
        Obtain all annotation labels for this document.

        Parameters
        ----------
        ann_filter
            A filter to apply to annotations, should map to annotations to ``True``
            if they should be included, ``False`` otherwise.

        Returns
        -------
        ``set[str]``
            A set containing all annotation labels for this document.
        """
        ann_filter = ann_filter or (lambda _: True)

        return {
            annotation.label
            for annotation in self.annotations
            if ann_filter(annotation)
        }

    def get_annotation_from_span(self, start: int, end: int) -> Annotation | None:
        """
        Get an annotation by span.

        Parameters
        ----------
        start
            The start char.
        end
            The end char.

        Returns
        -------
        ``Optional[Annotation]``
            The annotation with the provided span, or ``None`` if no such annotation
            exists.
        """
        for annotation in self.annotations:
            if (annotation.start == start) and (annotation.end == end):
                return annotation

        return None


@dataclass
class InfoExtractionDataset:
    """A dataset with annotated documents."""

    docs: list[Document]
    """The annotated documents."""

    _ALL_STATS: ClassVar[list] = [
        "num_docs",
        "num_annotations",
        "span_freqs",
        "label_freqs",
        "qualifier_freqs",
    ]

    @classmethod
    def from_clinlp_docs(
        cls,
        nlp_docs: Iterable[Doc],
        ids: Iterable[str] | None = None,
        spans_key: str = SPANS_KEY,
    ) -> "InfoExtractionDataset":
        """
        Create a dataset from ``clinlp`` documents.

        Parameters
        ----------
        nlp_docs
            An iterable of docs produced by ``clinlp`` (for example a list of ``Doc``,
            or a generator from ``nlp.pipe``).
        ids, optional
            An iterable of identifiers, that should have the same length as
            ``nlp_docs``. If not provided, will use a counter.
        spans_key
            The key in the ``Doc``'s ``.spans`` attribute where the annotations are
            stored.

        Returns
        -------
        ``InfoExtractionDataset``
            A dataset, corresponding to the provided ``clinlp`` documents.
        """
        ids = ids or itertools.count()

        docs = []

        for doc, identifier in zip(nlp_docs, ids, strict=False):
            annotations = [
                Annotation(
                    text=str(ent),
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                    qualifiers=ent._.qualifiers,
                )
                for ent in doc.spans[spans_key]
            ]

            docs.append(
                Document(
                    identifier=str(identifier), text=doc.text, annotations=annotations
                )
            )

        return cls(docs=docs)

    @classmethod
    def from_medcattrainer(
        cls,
        data: dict,
        *,
        strip_spans: bool = True,
    ) -> "InfoExtractionDataset":
        """
        Create a dataset from a ``MedCATTrainer`` export.

        Parameters
        ----------
        data
            The data from a ``MedCATTrainer`` export as a dictionary, as downloaded from
            the web interface in ``JSON`` format.
        strip_spans
            Whether to remove punctuation and whitespaces from the beginning or end
            of annotations. Used to clean up accidental over-annotations.

        Returns
        -------
        ``InfoExtractionDataset``
            A dataset, corresponding to the provided ``MedCATTrainer`` export.

        Raises
        ------
        ValueError
            If the ``MedCATTrainer`` export contains more than one project.
        """
        if len(data["projects"]) > 1:
            msg = "Cannot read MedCATTrainer exports with more than 1 project."
            raise ValueError(msg)

        data = data["projects"][0]

        docs = []

        for doc in data["documents"]:
            annotations = []

            for annotation in doc["annotations"]:
                if not annotation["deleted"]:
                    qualifiers = [
                        Qualifier(
                            name=qualifier["name"].title(),
                            value=qualifier["value"].title(),
                        )
                        for qualifier in annotation["meta_anns"].values()
                    ]

                    annotation = Annotation(
                        text=annotation["value"],
                        start=annotation["start"],
                        end=annotation["end"],
                        label=annotation["cui"],
                        qualifiers=qualifiers,
                    )

                    if strip_spans:
                        annotation.strip()

                    annotations.append(annotation)

            docs.append(
                Document(
                    identifier=doc["name"], text=doc["text"], annotations=annotations
                )
            )

        return cls(docs)

    @classmethod
    def from_dict(cls, data: dict) -> "InfoExtractionDataset":
        """
        Create a dataset from dictionary.

        Parameters
        ----------
        data
            The data in dictionary format.

        Returns
        -------
        ``InfoExtractionDataset``
            A dataset, corresponding to the provided dictionary data.
        """
        data = data.copy()

        for doc in data["docs"]:
            for ann in doc["annotations"]:
                if "qualifiers" in ann:
                    ann["qualifiers"] = [
                        Qualifier(**qualifier) for qualifier in ann.get("qualifiers")
                    ]

            doc["annotations"] = [Annotation(**ann) for ann in doc.get("annotations")]

        docs = [Document(**doc) for doc in data["docs"]]

        return cls(docs=docs)

    @classmethod
    def read_json(cls, file: str) -> "InfoExtractionDataset":
        """
        Read a dataset from a ``JSON`` file.

        Parameters
        ----------
        file
            The path to the file.

        Returns
        -------
        ``InfoExtractionDataset``
            A dataset, corresponding to the data in the provided file.
        """
        with pathlib.Path(file).open() as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_nervaluate(
        self, ann_filter: Callable[[Annotation], bool] | None = None
    ) -> list[list[dict]]:
        """
        Convert to ``nervaluate`` format.

        Parameters
        ----------
        ann_filter
            A filter to apply to annotations. Should map to annotations to ``True``
            if they should be included, ``False`` otherwise.

        Returns
        -------
        ``list[list[dict]]``
            A list of lists of dictionaries corresponding to annotations.
        """
        ann_filter = ann_filter or (lambda _: True)

        return [doc.to_nervaluate(ann_filter) for doc in self.docs]

    def to_dict(self) -> dict:
        """
        Convert to dictionary format.

        Returns
        -------
        ``dict``
            A dictionary with the items of this dataset.
        """
        return {"docs": [doc.to_dict() for doc in self.docs]}

    def write_json(self, file: str) -> None:
        """
        Write the dataset to a ``JSON`` file.

        Parameters
        ----------
        file
            The path to the file.
        """
        with pathlib.Path(file).open("w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def num_docs(self) -> int:
        """
        Compute the number of documents in this dataset.

        Returns
        -------
        ``int``
            The number of documents in this dataset.
        """
        return len(self.docs)

    def num_annotations(self) -> int:
        """
        Compute the number of annotations in all documents of this dataset.

        Returns
        -------
        ``int``
            The number of annotations in all documents of this dataset.
        """
        return sum(len(doc.annotations) for doc in self.docs)

    def span_freqs(
        self,
        n_spans: int | None = 25,
        span_callback: Callable | None = None,
    ) -> dict:
        """
        Compute frequency of all text spans in this dataset.

        Parameters
        ----------
        n_spans
            The ``n`` most frequent text spans to return.
        span_callback
            A callback applied to each text span. For instance useful for normalizing
            text.

        Returns
        -------
        ``dict``
            A dictionary containing the frequency of the requested text spans.
        """
        cntr = Counter()
        span_callback = span_callback or (lambda x: x)

        for doc in self.docs:
            cntr.update(
                [span_callback(annotation.text) for annotation in doc.annotations]
            )

        if n_spans is None:
            n_spans = len(cntr)

        return dict(cntr.most_common(n_spans))

    def label_freqs(
        self,
        n_labels: int | None = 25,
        label_callback: Callable | None = None,
    ) -> dict:
        """
        Compute frequency of all labels in this dataset.

        Parameters
        ----------
        n_spans
            The ``n`` most frequent labels to return.
        span_callback
            A callback applied to each label. For instance useful for normalizing text.

        Returns
        -------
        ``dict``
            A dictionary containing the frequency of the requested labels.
        """
        cntr = Counter()
        label_callback = label_callback or (lambda x: x)

        for doc in self.docs:
            cntr.update(
                [label_callback(annotation.label) for annotation in doc.annotations]
            )

        if n_labels is None:
            n_labels = len(cntr)

        return dict(cntr.most_common(n_labels))

    def qualifier_freqs(self) -> dict:
        """
        Compute the frequency of all qualifier values in this dataset.

        Returns
        -------
        ``dict``
            The computed frequencies, as a mapping from qualifier names to values to
            frequencies, e.g. ``{"Presence": {"Present": 25, "Absent": 10}, ...}``.
        """
        cntrs = defaultdict(lambda: Counter())

        for doc in self.docs:
            for annotation in doc.annotations:
                if annotation.qualifiers is not None:
                    for qualifier in annotation.qualifiers:
                        cntrs[qualifier.name].update([qualifier.value])

        return {name: dict(counts) for name, counts in cntrs.items()}

    def stats(self, **kwargs) -> dict:
        """
        Compute all statistics for this dataset.

        Combines the return values of all stats functions, defined in the ``_ALL_STATS``
        class variable. Any additional keyword arguments are passed to the respective
        stats functions, if they accept them.

        Returns
        -------
        ``dict``
            A dictionary containing all computed stats, e.g.
            ``{'num_docs': 384, 'num_annotations': 4353, ...}``.
        """
        stats = {}

        for stat in self._ALL_STATS:
            stat_func = getattr(self, stat)

            func_kwargs = {
                k: kwargs[k]
                for k in inspect.signature(stat_func).parameters
                if k in kwargs
            }

            stats[stat] = stat_func(**func_kwargs)

        return stats


class InfoExtractionMetrics:
    """Calculator for information extraction task metrics."""

    _QUALIFIER_METRICS: ClassVar[dict[str, Callable]] = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }

    def __init__(
        self, true: InfoExtractionDataset, pred: InfoExtractionDataset
    ) -> None:
        """
        Initialize information extraction metric calculator.

        Parameters
        ----------
        true
            The dataset containing true (annotated/gold standard) annotations.
        pred
            The dataset containing pred (predicted/inferred) annotations.
        """
        self.true = true
        self.pred = pred

        self._validate_dataset_compatibility()

    def _validate_dataset_compatibility(self) -> None:
        """
        Validate that the datasets are compatible for metric computation.

        Raises
        ------
        ValueError
            If the datasets don't have the same number of documents.
        ValueError
            If the datasets contain documents with non-matching identifiers.
        """
        if self.true.num_docs() != self.pred.num_docs():
            msg = "Can only compute metrics for Datasets with same size."
            raise ValueError(msg)

        for true_doc, pred_doc in zip(self.true.docs, self.pred.docs, strict=False):
            if true_doc.identifier != pred_doc.identifier:
                msg = (
                    "Found two documents with non-matching ids "
                    f"(true={true_doc.identifier}, pred={pred_doc.identifier}). "
                    "Please make sure to present the same documents, "
                    "in the same order."
                )

                raise ValueError(msg)

    def entity_metrics(
        self,
        *,
        ann_filter: Callable[[Annotation], bool] | None = None,
        per_label: bool = False,
    ) -> dict:
        """
        Compute metrics for entities, including precision, recall and f1-score.

        Computes measures for exact, strict, partial, and type matching, using the
        ``[nervaluate](https://github.com/MantisAI/nervaluate)`` implementation of
        the SemEval 2013 9.1 task evaluation.

        Parameters
        ----------
        ann_filter
            A filter to apply to annotations. Can for instance be used to exclude
            annotations with certain labels or qualifiers. Annotations are only
            included in the metrics computation if the filter maps them to ``True``.
        per_label
            Whether to compute metrics per label. If set to ``True``, will
            micro-average the metrics across all labels.

        Returns
        -------
        ``dict``
            The computed entity metrics.
        """
        ann_filter = ann_filter or (lambda _: True)

        true_anns = self.true.to_nervaluate(ann_filter)
        pred_anns = self.pred.to_nervaluate(ann_filter)

        labels = list(
            set.union(
                *[doc.labels(ann_filter) for doc in self.true.docs + self.pred.docs]
            )
        )

        evaluator = nervaluate.Evaluator(true=true_anns, pred=pred_anns, tags=labels)

        results, class_results = evaluator.evaluate()

        return class_results if per_label else results

    def _aggregate_qualifier_values(self) -> dict[str, dict[str, list]]:
        """
        Aggregate qualifier values for true and predicted annotations.

        Matches annotations based on their start and end char, and only includes
        annotations with the same start and end char in the aggregation. Only includes
        qualifiers that are present in both the true and predicted annotations.

        Returns
        -------
        ``dict[str, dict[str, list]]``
            A dictionary containing the aggregated qualifier values, e.g.:

            ```
            {
                "Presence": {
                    "true": ["Present", "Absent", "Present"],
                    "pred": ["Present", "Absent", "Absent"],
                    "misses": [
                        {"doc.identifier": 1, annotation: {"start": 0, "end": 5, "text":
                        "test"}, true_qualifier: "Present", pred_qualifier: "Absent"},
                        ...]
                },
                ...
            }
            ```
        """
        aggregation: dict = defaultdict(lambda: defaultdict(list))

        for true_doc, pred_doc in zip(self.true.docs, self.pred.docs, strict=False):
            for true_annotation in true_doc.annotations:
                pred_annotation = pred_doc.get_annotation_from_span(
                    start=true_annotation.start, end=true_annotation.end
                )

                if pred_annotation is None:
                    continue

                qualifier_names = true_annotation.qualifier_names.intersection(
                    pred_annotation.qualifier_names
                )

                for name in qualifier_names:
                    true_value = true_annotation.get_qualifier_by_name(name).value
                    pred_value = pred_annotation.get_qualifier_by_name(name).value

                    aggregation[name]["true"].append(true_value)
                    aggregation[name]["pred"].append(pred_value)

                    if true_value != pred_value:
                        aggregation[name]["misses"].append(
                            {
                                "doc.identifier": true_doc.identifier,
                                "annotation": true_annotation.to_nervaluate(),
                                "true_qualifier": true_value,
                                "pred_qualifier": pred_value,
                            }
                        )

        return aggregation

    def qualifier_metrics(self, *, misses: bool = True) -> dict:
        """
        Compute metrics for qualifiers, including precision, recall and f1-score.

        Parameters
        ----------
        misses
            Whether to include all misses (false positives/negatives) in the results.

        Returns
        -------
        ``dict``
            The computed qualifier metrics.

        Raises
        ------
        ValueError
            If the datasets contain non-binary qualifier values.
        """
        aggregation = self._aggregate_qualifier_values()

        result = {}

        for name, values in aggregation.items():
            result[name] = {"metrics": {"n": len(values["true"])}}

            if misses:
                result[name]["misses"] = values["misses"]

            metrics = InfoExtractionMetrics._QUALIFIER_METRICS

            for metric_name, metric_func in metrics.items():
                metric_result = metric_func(
                    values["true"], values["pred"], average="micro"
                )

                result[name]["metrics"][metric_name] = metric_result

        return result
