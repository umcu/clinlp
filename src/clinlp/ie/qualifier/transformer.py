"""Transformer-based qualifier detectors."""

import statistics
from collections.abc import Callable

import numpy as np
import torch
from spacy import Language
from spacy.tokens import Doc, Span
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    RobertaForTokenClassification,
)

from clinlp.ie.qualifier.qualifier import (
    ATTR_QUALIFIERS,
    QualifierClass,
    QualifierDetector,
)
from clinlp.util import clinlp_component

class QualifierTransformer(QualifierDetector):
    """
    Transformer-based qualifier detector.

    Implements helper methods and lazy-loading for transformer-based qualifier
    detectors.
    """

    PRETRAINED_MODEL_NAME_OR_PATH: str
    REVISION: str
    MODEL_CLASS = AutoModelForTokenClassification
    TRUST_REMOTE_CODE = False

    _tokenizer: PreTrainedTokenizerBase | None = None
    _model: PreTrainedModel | None = None

    def __init__(
        self,
        *,
        token_window: int = 32,
        strip_entities: bool = True,
        placeholder: str | None = None,
        prob_aggregator: Callable = statistics.mean,
        **kwargs,
    ) -> None:
        """
        Create a transformer-based qualifier detector.

        Parameters
        ----------
        token_window
            The number of tokens to include before and after an entity.
        strip_entities
            Whether to strip whitespaces etc. from entities from the text.
        placeholder
            The placeholder to replace the entity with intext.
        prob_aggregator
            The function to aggregate the probabilities of the tokens in the entity.
        """
        self.token_window = token_window
        self.strip_entities = strip_entities
        self.placeholder = placeholder
        self.prob_aggregator = prob_aggregator

        super().__init__(**kwargs)

    @classmethod
    def _load_pretrained(
        cls, *, force_download: bool = False
    ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """Load and cache the pretrained tokenizer and model for this component."""
        if cls._tokenizer is None or force_download:
            cls._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=cls.PRETRAINED_MODEL_NAME_OR_PATH,
                revision=cls.REVISION,
                force_download=force_download,
                trust_remote_code=cls.TRUST_REMOTE_CODE,
            )

        if cls._model is None or force_download:
            cls._model = cls.MODEL_CLASS.from_pretrained(
                pretrained_model_name_or_path=cls.PRETRAINED_MODEL_NAME_OR_PATH,
                revision=cls.REVISION,
                force_download=force_download,
                trust_remote_code=cls.TRUST_REMOTE_CODE,
            )

        return cls._tokenizer, cls._model

    @classmethod
    def prefetch(cls, *, force_download: bool = False) -> None:
        """Pre-download all model assets used by this transformer."""
        cls._load_pretrained(force_download=force_download)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer."""
        tokenizer, _ = self.__class__._load_pretrained()
        return tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """The model."""
        _, model = self.__class__._load_pretrained()
        return model

    @staticmethod
    def _get_ent_window(ent: Span, token_window: int) -> tuple[str, int, int]:
        """
        Get the entity window.

        The window includes the tokens of the entity itself, and a number of tokens
        before and after the entity.

        Parameters
        ----------
        ent
            The entity.
        token_window
            The number of tokens to include before and after the entity.

        Returns
        -------
        ``str``
            The text span based on the window.
        ``int``
            The original entity start character.
        ``int``
            The original entity end character.
        """
        start_token_i = max(0, ent.start - token_window)
        end_token_i = min(len(ent.doc), ent.end + token_window)

        text_span = ent.doc[start_token_i:end_token_i]

        ent_start_char = ent.start_char - text_span.start_char
        ent_end_char = ent.end_char - text_span.start_char

        return str(text_span), ent_start_char, ent_end_char

    @staticmethod
    def _trim_ent_boundaries(
        text: str, ent_start_char: int, ent_end_char: int
    ) -> tuple[str, int, int]:
        """
        Trim the boundaries of an entity.

        Parameters
        ----------
        text
            The text.
        ent_start_char
            The entity start character.
        ent_end_char
            The entity end character.

        Returns
        -------
        ``str``
            The modified text text.
        ``int``
            The entity start character.
        ``int``
            The entity end character.
        """
        entity = text[ent_start_char:ent_end_char]

        ent_start_char += len(entity) - len(entity.lstrip())
        ent_end_char -= len(entity) - len(entity.rstrip())

        return text, ent_start_char, ent_end_char

    @staticmethod
    def _fill_ent_placeholder(
        text: str, ent_start_char: int, ent_end_char: int, placeholder: str
    ) -> tuple[str, int, int]:
        """
        Replace the entity intext with a placeholder.

        Parameters
        ----------
        text
            The text.
        ent_start_char
            The entity start character.
        ent_end_char
            The entity end character.
        placeholder
            The placeholder.

        Returns
        -------
        ``str``
            The modified text.
        ``int``
            The entity start character.
        ``int``
            The entity end character.
        """
        text = text[0:ent_start_char] + placeholder + text[ent_end_char:]
        ent_end_char = ent_start_char + len(placeholder)

        return text, ent_start_char, ent_end_char

    def _prepare_ent(self, ent: Span) -> tuple[str, int, int]:
        """
        Prepare the entity for prediction.

        Applies the configured settings to the entity.

        Parameters
        ----------
        ent
            The entity.

        Returns
        -------
        ``str``
            The modified text.
        ``int``
            The entity start character.
        ``int``
            The entity end character.
        """
        text, ent_start_char, ent_end_char = self._get_ent_window(
            ent, token_window=self.token_window
        )

        if self.strip_entities:
            text, ent_start_char, ent_end_char = self._trim_ent_boundaries(
                text, ent_start_char, ent_end_char
            )

        if self.placeholder is not None:
            text, ent_start_char, ent_end_char = self._fill_ent_placeholder(
                text, ent_start_char, ent_end_char, placeholder=self.placeholder
            )

        return text, ent_start_char, ent_end_char

    @staticmethod
    def _get_ent_token_span(
        inputs, ent_start_char: int, ent_end_char: int
    ) -> tuple[int, int]:
        """Map character span to token span for transformer inputs."""
        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1)

        if start_token is not None and end_token is not None:
            return start_token, end_token

        overlapping_token_indices = [
            token_i
            for token_i, (token_start, token_end) in enumerate(inputs.encodings[0].offsets)
            if token_start < ent_end_char and token_end > ent_start_char
        ]

        if len(overlapping_token_indices) == 0:
            msg = (
                "Could not align entity character span to token span: "
                f"[{ent_start_char}:{ent_end_char}]"
            )
            raise ValueError(msg)

        return overlapping_token_indices[0], overlapping_token_indices[-1]

    def _predict(
        self,
        text: str,
        ent_start_char: int,
        ent_end_char: int,
        prob_indices: list,
        prob_aggregator: Callable,
    ) -> float:
        """
        Predict the probability of a qualifier.

        Parameters
        ----------
        text
            The text.
        ent_start_char
            The entity start character.
        ent_end_char
            The entity end character.
        prob_indices
            The indices of the probabilities to aggregate.
        prob_aggregator
            The function to aggregate the probabilities.

        Returns
        -------
        ``float``
            The probability.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probs = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token, end_token = self._get_ent_token_span(
            inputs, ent_start_char, ent_end_char
        )

        return prob_aggregator(
            sum(pos[prob_indices]) for pos in probs[start_token : end_token + 1]
        )

    def _predict_multiclass(
        self,
        text: str,
        ent_start_char: int,
        ent_end_char: int,
        qualifier_mapping: dict[str, str] | None = None,
    ) -> tuple[str, float]:
        """
        Predict the probability of a multiclass qualifier.

        Parameters
        ----------
        text
            The text.
        ent_start_char
            The entity start character.
        ent_end_char
            The entity end character.

        qualifier_mapping
            Optional mapping from model output labels to reduced labels. When
            provided, probabilities of labels that map to the same reduced label
            are averaged before selecting the final prediction.

        Returns
        -------
        ``tuple``
            The qualifier (str), The probability (float)
        """

        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probs = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token, end_token = self._get_ent_token_span(
            inputs, ent_start_char, ent_end_char
        )

        mean_probas = np.mean(probs[start_token : end_token + 1, :], axis=0)

        if isinstance(qualifier_mapping, dict):
            aggregated_probas: dict[str, float] = {}
            aggregated_counts: dict[str, int] = {}

            for label_id, label_proba in enumerate(mean_probas):
                label = (
                    self.model.config.id2label[label_id]
                    .removeprefix("B-")
                    .removeprefix("I-")
                )
                reduced_label = qualifier_mapping.get(label, label)

                aggregated_probas[reduced_label] = (
                    aggregated_probas.get(reduced_label, 0.0) + float(label_proba)
                )
                aggregated_counts[reduced_label] = aggregated_counts.get(reduced_label, 0) + 1

            aggregated_mean_probas = {
                reduced_label: aggregated_probas[reduced_label]
                / aggregated_counts[reduced_label]
                for reduced_label in aggregated_probas
            }

            qualifier, probability = max(
                aggregated_mean_probas.items(),
                key=lambda item: item[1],
            )
            return qualifier, probability

        label_id = int(np.argmax(mean_probas))
        qualifier = self.model.config.id2label[label_id].removeprefix("B-").removeprefix("I-")
        probability = float(mean_probas[label_id])

        return qualifier, probability



@clinlp_component(
    name="clinlp_negation_transformer",
    requires=["doc.spans"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
)
class NegationTransformer(QualifierTransformer):
    """Transformer-based negation detector."""

    PRETRAINED_MODEL_NAME_OR_PATH = "UMCU/MedRoBERTa.nl_NegationDetection"
    REVISION = "83068ba132b6ce38e9f668c1e3ab636f79b774d3"
    MODEL_CLASS = RobertaForTokenClassification

    def __init__(
        self,
        nlp: Language,
        *,
        absence_threshold: float = 0.9,
        presence_threshold: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Create a transformer-based negation detector.

        Parameters
        ----------
        nlp
            The ``spaCy`` language model.
        absence_threshold
            The threshold for absence. Will classify qualifier as ``Presence.Absent``
            if ``prediction`` < ``absence_threshold``.
        presence_threshold
            The threshold for presence. Will classify qualifier as ``Presence.Present``
            if ``prediction`` > ``presence_threshold``.
        """
        self.nlp = nlp
        self.absence_threshold = absence_threshold
        self.presence_threshold = presence_threshold

        super().__init__(**kwargs)

    @property
    def qualifier_classes(self) -> dict[str, QualifierClass]:  # noqa: D102
        return {
            "Presence": QualifierClass(
                "Presence", ["Absent", "Uncertain", "Present"], default="Present"
            )
        }

    def _detect_qualifiers(self, doc: Doc) -> None:
        """
        Detect qualifiers for the entities in a document.

        Prepares the entity, then predicts the probability of the qualifier. If the
        probability is below the absence threshold, the qualifier is classified as
        "Absent". If the probability is above the presence threshold, the qualifier is
        classified as "Present". Otherwise, it is classified as "Uncertain".

        Parameters
        ----------
        doc
            The document to process.
        """
        for ent in doc.spans[self.spans_key]:
            text, ent_start_char, ent_end_char = self._prepare_ent(ent)

            prob = 1 - self._predict(
                text,
                ent_start_char,
                ent_end_char,
                prob_indices=[0, 2],
                prob_aggregator=self.prob_aggregator,
            )

            if prob <= self.absence_threshold:
                qualifier_value = "Absent"
            elif prob >= self.presence_threshold:
                qualifier_value = "Present"
            else:
                qualifier_value = "Uncertain"

            self.add_qualifier_to_ent(
                ent,
                self.qualifier_classes["Presence"].create(qualifier_value, prob=prob),
            )


@clinlp_component(
    name="clinlp_experiencer_transformer",
    requires=["doc.spans"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
)
class ExperiencerTransformer(QualifierTransformer):
    """
    Transformer-based experiencer detector.

    Currently, only detects ``Experiencer.Patient`` (default) and ``Experiencer.Family``
    -- ``Experiencer.Other`` is not yet implemented.
    """

    PRETRAINED_MODEL_NAME_OR_PATH = "UMCU/MedRoBERTa.nl_Experiencer"
    REVISION = "d9318c4b2b0ab0dfe50afedca58319b2369f1a71"
    MODEL_CLASS = RobertaForTokenClassification

    def __init__(
        self,
        nlp: Language,
        *,
        family_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        """
        Create a transformer-based experiencer detector.

        Parameters
        ----------
        nlp
            The ``spaCy`` language model.
        family_threshold
            The threshold for family. Will classify qualifier as ``Experiencer.Family``
            if ``prediction`` > ``family_threshold``.
        """
        self.nlp = nlp
        self.family_threshold = family_threshold

        super().__init__(**kwargs)

    @property
    def qualifier_classes(self) -> dict[str, QualifierClass]:  # noqa: D102
        return {
            "Experiencer": QualifierClass(
                "Experiencer", ["Patient", "Family", "Other"], default="Patient"
            )
        }

    def _detect_qualifiers(self, doc: Doc) -> None:
        """
        Detect qualifiers for the entities in a document.

        Prepares the entity, then predicts the probability of the qualifier. If the
        probability is above the family threshold, the qualifier is classified as
        "Family". Otherwise, it is classified as "Patient" (default).

        Parameters
        ----------
        doc
            The document to process.
        """
        for ent in doc.spans[self.spans_key]:
            text, ent_start_char, ent_end_char = self._prepare_ent(ent)

            prob = self._predict(
                text,
                ent_start_char,
                ent_end_char,
                prob_indices=[1, 3],
                prob_aggregator=self.prob_aggregator,
            )

            if prob > self.family_threshold:
                self.add_qualifier_to_ent(
                    ent,
                    self.qualifier_classes["Experiencer"].create("Family", prob=prob),
                )


@clinlp_component(
    name="clinlp_pii_transformer",
    requires=["doc.spans"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
)
class PiiTransformer(QualifierTransformer):
    """Transformer-based PII detector."""

    PRETRAINED_MODEL_NAME_OR_PATH = "UMCU/PII_RobBERT"
    REVISION = "7cae7b32d565fc49430f08517c5e8dbc20382478"
    TRUST_REMOTE_CODE = True

    def __init__(
        self,
        nlp: Language,
        *,
        proba_threshold: float = 0.1,
        reduce_labels: bool = True,
        **kwargs,
    ) -> None:
        """
        Create a transformer-based PII detector.

        Parameters
        ----------
        nlp
            The ``spaCy`` language model.
        proba_threshold
            The threshold for any detection. Will classify qualifier as ``PII.Absent``
            if ``prediction`` < ``proba_threshold``.
        reduce_labels
            Whether to reduce the number of labels to a more generic set.
        """
        self.nlp = nlp
        self.proba_threshold = proba_threshold
        self.qualifier_mapping = {}

        if reduce_labels:
            self.qualifier_mapping = {
                "GIVENNAME": "NAME",
                "SURNAME": "NAME",
                "TAXNUM": "IDNUM",
                "DRIVERLICENSENUM": "IDNUM",
                "CREDITCARDNUMBER": "IDNUM",
                "PASSPORTNUM": "IDNUM",
                "SOCIALNUM": "IDNUM",
                "BUILDINGNUM": "LOCATION",
                "CITY": "LOCATION",
                "STREET": "LOCATION",
                "ZIPCODE": "LOCATION",
                "GENDER": "SEX",
            }
            # TODO: redundant once tested?
            assert all(
                [
                    k
                    in [
                        _k.removeprefix("B-").removeprefix("I-")
                        for _k in self.model.config.label2id.keys()
                    ]
                    for k in self.qualifier_mapping.keys()
                ]
            ), "Not all mapping keys map to an actual label"

        super().__init__(**kwargs)

    @property
    def qualifier_classes(self) -> dict[str, QualifierClass]:  # noqa: D102
        return {
            "PII": QualifierClass(
                "PII",
                [
                    "Absent",
                    "AGE",
                    "BUILDINGNUM",
                    "CITY",
                    "CREDITCARDNUMBER",
                    "DATE",
                    "DRIVERLICENSENUM",
                    "EMAIL",
                    "GENDER",
                    "GIVENNAME",
                    "IDCARDNUM",
                    "IDNUM",
                    "LOCATION",
                    "NAME",
                    "PASSPORTNUM",
                    "SEX",
                    "SOCIALNUM",
                    "STREET",
                    "SURNAME",
                    "TAXNUM",
                    "TELEPHONENUM",
                    "TITLE",
                    "ZIPCODE",
                ],
                default="Absent",
            )
        }

    def _detect_qualifiers(self, doc: Doc) -> None:
        """
        Detect qualifiers for the entities in a document.

        Prepares the entity, then predicts the probability of the qualifier. If the
        probability is below the absence threshold, the qualifier is classified as
        "Absent". If the probability is above the presence threshold, the qualifier is
        classified as "Present". Otherwise, it is classified as "Uncertain".

        Parameters
        ----------
        doc
            The document to process.
        """
        for ent in doc.spans[self.spans_key]:
            text, ent_start_char, ent_end_char = self._prepare_ent(ent)

            qualifier, prob = self._predict_multiclass(
                text,
                ent_start_char,
                ent_end_char,
                self.qualifier_mapping,
            )

            if (
                prob <= self.proba_threshold
                or qualifier not in self.qualifier_classes["PII"].values
            ):
                qualifier_value = "Absent"
            else:
                if qualifier == "O":
                    qualifier_value = "Absent"
                else:
                    qualifier_value = qualifier

            self.add_qualifier_to_ent(
                ent,
                self.qualifier_classes["PII"].create(qualifier_value, prob=prob),
            )

@clinlp_component(
    name="clinlp_cardioner_transformer",
    requires=["doc.spans"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
)
class CardioTransformer(QualifierTransformer):
    """Transformer-based detector of medication/procedure/symptom/disease. For the cardiology domain."""

    PRETRAINED_MODEL_NAME_OR_PATH = "DT4H/cardio-ner-multilingual-xlm-roberta-large-multilabel"
    REVISION = "8dd428df5af2df36a62544c3020c0b15d617227c"

    def __init__(
        self,
        nlp: Language,
        *,
        proba_threshold: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Create a transformer-based medical entity detector.

        Parameters
        ----------
        nlp
            The ``spaCy`` language model.
        proba_threshold
            The threshold for any detection. Will classify qualifier as ``MED_ENT.Absent``
            if ``prediction`` < ``proba_threshold``.
        """
        self.nlp = nlp
        self.proba_threshold = proba_threshold
        self.qualifier_mapping = {}

        super().__init__(**kwargs)

    @property
    def qualifier_classes(self) -> dict[str, QualifierClass]:  # noqa: D102
        return {
            "MED_NER": QualifierClass(
                "MED_NER",
                [
                    "Absent",
                    "DISEASE",
                    "SYMPTOM",
                    "MEDICATION",
                    "PROCEDURE"
                ],
                default="Absent",
            )
        }

    def _detect_qualifiers(self, doc: Doc) -> None:
        """
        Detect qualifiers for the entities in a document.

        Prepares the entity, then predicts the probability of the qualifier. If the
        probability is below the absence threshold, the qualifier is classified as
        "Absent". If the probability is above the presence threshold, the qualifier is
        classified as "Present". Otherwise, it is classified as "Uncertain".

        Parameters
        ----------
        doc
            The document to process.
        """
        for ent in doc.spans[self.spans_key]:
            text, ent_start_char, ent_end_char = self._prepare_ent(ent)

            qualifier, prob = self._predict_multiclass(
                text,
                ent_start_char,
                ent_end_char,
            )

            if (
                prob <= self.proba_threshold
                or qualifier not in self.qualifier_classes["MED_NER"].values
            ):
                qualifier_value = "Absent"
            else:
                if qualifier == "O":
                    qualifier_value = "Absent"
                else:
                    qualifier_value = qualifier

            self.add_qualifier_to_ent(
                ent,
                self.qualifier_classes["MED_NER"].create(qualifier_value, prob=prob),
            )
