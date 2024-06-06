"""Classes and functions for evaluating information extraction tasks."""

import inspect
import itertools
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterable, Optional

import nervaluate
from sklearn.metrics import f1_score, precision_score, recall_score
from spacy.language import Doc

from clinlp.ie import SPANS_KEY


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

    qualifiers: list[dict] = field(default_factory=list)
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
        Convert to format that ``nervaluate`` ingests.

        Returns
        -------
            A dictionary with the items ``nervaluate`` expects.
        """
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "label": self.label,
        }

    @property
    def qualifier_names(self) -> set[str]:
        """
        Obtain unique qualifier names for this annotation.

        Returns
        -------
            A set of unique qualifier names, e.g. {"Presence", "Experiencer"}.
        """
        return {qualifier["name"] for qualifier in self.qualifiers}

    def get_qualifier_by_name(self, qualifier_name: str) -> dict:
        """
        Get a qualifier by name.

        Parameters
        ----------
        qualifier_name
            The name of the qualifier.

        Returns
        -------
            The qualifier with the provided name.

        Raises
        ------
        KeyError
            If no qualifier with the provided name exists.
        """
        for qualifier in self.qualifiers:
            if qualifier["name"] == qualifier_name:
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
        self, ann_filter: Optional[Callable[[Annotation], bool]] = None
    ) -> list[dict]:
        """
        Convert to format that ``nervaluate`` ingests.

        Parameters
        ----------
        ann_filter
            A filter to apply to annotations. Should map the annotations to ``True``
            if they should be included, ``False`` otherwise.

        Returns
        -------
            A list of dictionaries corresponding to annotations.
        """
        ann_filter = ann_filter or (lambda _: True)

        return [ann.to_nervaluate() for ann in self.annotations if ann_filter(ann)]

    def labels(
        self, ann_filter: Optional[Callable[[Annotation], bool]] = None
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
            A set containing all annotation labels for this document.
        """
        ann_filter = ann_filter or (lambda _: True)

        return {
            annotation.label
            for annotation in self.annotations
            if ann_filter(annotation)
        }

    def get_annotation_from_span(self, start: int, end: int) -> Optional[Annotation]:
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

    default_qualifiers: Optional[dict[str, str]] = None
    """
    Mapping of qualifiers to their default value, e.g.
    ``{"Presence": "Present"}``.
    """

    _ALL_STATS: ClassVar[list] = [
        "num_docs",
        "num_annotations",
        "span_freqs",
        "label_freqs",
        "qualifier_freqs",
    ]

    def __post_init__(self) -> None:
        """
        Initialize the dataset.

        Initializes the default qualifiers, from the annotations (if available) or
        infers them from the majority class.
        """
        self.default_qualifiers = {}

        try:
            for doc in self.docs:
                for annotation in doc.annotations:
                    for qualifier in annotation.qualifiers:
                        if qualifier["is_default"]:
                            self.default_qualifiers[qualifier["name"]] = qualifier[
                                "value"
                            ]
        except KeyError:
            self.default_qualifiers = self.infer_default_qualifiers()

    def infer_default_qualifiers(self) -> dict:
        """
        Infer and set Annotations' default qualifiers from majority classes.

        Returns
        -------
            A dictionary mapping qualifier names to their default values.
        """
        default_qualifiers = {
            name: max(counts, key=lambda item: counts[item])
            for name, counts in self.qualifier_freqs().items()
        }

        warnings.warn(
            f"Inferred the following qualifier defaults from the majority "
            f"classes: {default_qualifiers}. ",
            UserWarning,
            stacklevel=2,
        )

        for doc in self.docs:
            for annotation in doc.annotations:
                for qualifier in annotation.qualifiers:
                    qualifier["is_default"] = (
                        default_qualifiers[qualifier["name"]] == qualifier["value"]
                    )

        return default_qualifiers

    @staticmethod
    def from_clinlp_docs(
        nlp_docs: Iterable[Doc], ids: Optional[Iterable[str]] = None
    ) -> "InfoExtractionDataset":
        """
        Create a new dataset from ``clinlp`` documents.

        Parameters
        ----------
        nlp_docs
            An iterable of docs produced by ``clinlp`` (for example a list of ``Doc``,
            or a generator from ``nlp.pipe``)
        ids, optional
            An iterable of identifiers, that should have the same length as
            ``nlp_docs``. If not provided, will use a counter.

        Returns
        -------
            A dataset, corresponding to the provided ``clinlp`` documents.
        """
        ids = ids or itertools.count()

        docs = []

        for doc, identifier in zip(nlp_docs, ids):
            annotations = []

            for ent in doc.spans[SPANS_KEY]:
                qualifiers = [
                    {
                        "name": qualifier.name.title(),
                        "value": qualifier.value.title(),
                        "is_default": qualifier.is_default,
                    }
                    for qualifier in ent._.qualifiers
                ]

                annotations.append(
                    Annotation(
                        text=str(ent),
                        start=ent.start_char,
                        end=ent.end_char,
                        label=ent.label_,
                        qualifiers=qualifiers,
                    )
                )

            docs.append(
                Document(
                    identifier=str(identifier), text=doc.text, annotations=annotations
                )
            )

        return InfoExtractionDataset(docs=docs)

    @staticmethod
    def from_medcattrainer(
        data: dict,
        *,
        strip_spans: bool = True,
        default_qualifiers: Optional[dict[str, str]] = None,
    ) -> "InfoExtractionDataset":
        """
        Create a new dataset from a ``MedCATTrainer`` export.

        Parameters
        ----------
        data
            The data from a ``MedCATTrainer`` export as a dictionary, as downloaded from
            the web interface in ``JSON`` format.
        strip_spans
            Whether to remove punctuation and whitespaces from the beginning or end
            of annotations. Used to clean up accidental over-annotations.
        default_qualifiers
            The default qualifiers (which are not included in the ``MedCATTrainer``
            export), e.g. ``{"Presence": "Absent", "Experiencer": "Patient"}``, by
            default ``None``. If ``None``, will infer the default qualifiers from the
            majority class.

        Returns
        -------
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
                    qualifiers = []

                    for qualifier in annotation["meta_anns"].values():
                        qualifier = {
                            "name": qualifier["name"].title(),
                            "value": qualifier["value"].title(),
                        }

                        if default_qualifiers is not None:
                            qualifier["is_default"] = (
                                default_qualifiers[qualifier["name"]]
                                == qualifier["value"]
                            )

                        qualifiers.append(qualifier)

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

        return InfoExtractionDataset(docs)

    def to_nervaluate(
        self, ann_filter: Optional[Callable[[Annotation], bool]] = None
    ) -> list[list[dict]]:
        """
        Convert to format that ``nervaluate`` ingests.

        Parameters
        ----------
        ann_filter
            A filter to apply to annotations. Should map to annotations to ``True``
            if they should be included, ``False`` otherwise.

        Returns
        -------
            A list of lists of dictionaries corresponding to annotations.
        """
        ann_filter = ann_filter or (lambda _: True)

        return [doc.to_nervaluate(ann_filter) for doc in self.docs]

    def num_docs(self) -> int:
        """
        Compute the number of documents in this dataset.

        Returns
        -------
            The number of documents in this dataset.
        """
        return len(self.docs)

    def num_annotations(self) -> int:
        """
        Compute the number of annotations in all documents of this dataset.

        Returns
        -------
            The number of annotations in all documents of this dataset.
        """
        return sum(len(doc.annotations) for doc in self.docs)

    def span_freqs(
        self,
        n_spans: Optional[int] = 25,
        span_callback: Optional[Callable] = None,
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
        n_labels: Optional[int] = 25,
        label_callback: Optional[Callable] = None,
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
        Compute frequency of all qualifier values in this dataset.

        Returns
        -------
            The computed frequencies, as a mapping from qualifier names to values to
            frequencies, e.g. ``{"Presence": {"Present": 25, "Absent": 10}, ...}``.
        """
        cntrs = defaultdict(lambda: Counter())

        for doc in self.docs:
            for annotation in doc.annotations:
                for qualifier in annotation.qualifiers:
                    cntrs[qualifier["name"]].update([qualifier["value"]])

        return {name: dict(counts) for name, counts in cntrs.items()}

    def stats(self, **kwargs) -> dict:
        """
        Compute all stats for this dataset.

        Combines the return values of all stats functions, defined in the ``_ALL_STATS``
        class variable. Any additional keyword arguments are passed to the respective
        stats functions, if they accept them.

        Returns
        -------
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
        Create a new metric computation instance.

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
            msg = "Can only compute metrics for Datasets with same size"
            raise ValueError(msg)

        for true_doc, pred_doc in zip(self.true.docs, self.pred.docs):
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
        ann_filter: Optional[Callable[[Annotation], bool]] = None,
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
            A dictionary containing the aggregated qualifier values, e.g.:

            ```
            {
                "Presence": {
                    "true": ["Present", "Absent", "Present"],
                    "pred": ["Present", "Absent", "Absent"],
                    "misses": [
                        {"doc.identifier": 1, annotation: {"start": 0, "end": 5, "text":
                        "test"}, true_label: "Present", pred_label: "Absent"}, ...]
                },
                ...
            }
            ```
        """
        aggregation: dict = defaultdict(lambda: defaultdict(list))

        for true_doc, pred_doc in zip(self.true.docs, self.pred.docs):
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
                    true_val = true_annotation.get_qualifier_by_name(name)["value"]
                    pred_val = pred_annotation.get_qualifier_by_name(name)["value"]

                    aggregation[name]["true"].append(true_val)
                    aggregation[name]["pred"].append(pred_val)

                    if true_val != pred_val:
                        aggregation[name]["misses"].append(
                            {
                                "doc.identifier": true_doc.identifier,
                                "annotation": true_annotation.to_nervaluate(),
                                "true_qualifier": true_val,
                                "pred_qualifier": pred_val,
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
            The computed qualifier metrics.

        Raises
        ------
        ValueError
            If the datasets contain non-binary qualifier values.
        """
        aggregation = self._aggregate_qualifier_values()

        result = {}

        for name, values in aggregation.items():
            true_unique_values = set(values["true"])
            pred_unique_values = set(values["pred"])

            if max(len(true_unique_values), len(pred_unique_values)) > 2:
                msg = "Can oly compute metrics for binary qualifier values"
                raise ValueError(msg)

            pos_label = next(
                val
                for val in true_unique_values
                if val != self.true.default_qualifiers[name]
            )

            result[name] = {
                "metrics": {
                    "n": len(values["true"]),
                    "n_pos_true": sum(1 for v in values["true"] if v == pos_label),
                    "n_pos_pred": sum(1 for v in values["pred"] if v == pos_label),
                },
            }

            if misses:
                result[name]["misses"] = values["misses"]

            for (
                metric_name,
                metric_func,
            ) in InfoExtractionMetrics._QUALIFIER_METRICS.items():
                result[name]["metrics"][metric_name] = metric_func(
                    values["true"], values["pred"], pos_label=pos_label
                )

        return result
