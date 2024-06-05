"""Transformer-based qualifier detectors."""

import statistics
from abc import abstractmethod
from typing import Callable, Tuple

import torch
from spacy import Language
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, RobertaForTokenClassification

from clinlp.ie.qualifier.qualifier import (
    ATTR_QUALIFIERS,
    QualifierClass,
    QualifierDetector,
)
from clinlp.util import clinlp_component

_defaults_qualifier_transformer = {
    "token_window": 32,
    "strip_entities": True,
    "placeholder": None,
    "prob_aggregator": statistics.mean,
}

_defaults_negation_transformer = {
    "absence_threshold": 0.1,
    "presence_threshold": 0.9,
}
_defaults_experiencer_transformer = {
    "token_window": 64,
    "family_threshold": 0.5,
}


class QualifierTransformer(QualifierDetector):
    """
    Transformer-based qualifier detector.

    Implements some helper methods, but cannot be used directly. Specifically, does not
    implement the abstract properties ``qualifier_classes``, ``tokenizer`` and 
    ``model``, and abstract method ``_detect_qualifiers``.

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

    def __init__(
        self,
        token_window: int = _defaults_qualifier_transformer["token_window"],
        strip_entities: bool = _defaults_qualifier_transformer["strip_entities"],  # noqa: FBT001
        placeholder: str = _defaults_qualifier_transformer["placeholder"],
        prob_aggregator: Callable = _defaults_qualifier_transformer["prob_aggregator"],
        **kwargs,
    ) -> None:
        self.token_window = token_window
        self.strip_entities = strip_entities
        self.placeholder = placeholder
        self.prob_aggregator = prob_aggregator

        super().__init__(**kwargs)

    @property
    @abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        """
        The tokenizer.

        Returns
        -------
            The tokenizer.
        """

    @property
    @abstractmethod
    def model(self) -> RobertaForTokenClassification:
        """
        The model.

        Returns
        -------
            The model.
        """

    @staticmethod
    def _get_ent_window(ent: Span, token_window: int) -> Tuple[str, int, int]:
        start_token_i = max(0, ent.start - token_window)
        end_token_i = min(len(ent.doc), ent.end + token_window)

        text_span = ent.doc[start_token_i:end_token_i]

        ent_start_char = ent.start_char - text_span.start_char
        ent_end_char = ent.end_char - text_span.start_char

        return str(text_span), ent_start_char, ent_end_char

    @staticmethod
    def _trim_ent_boundaries(
        text: str, ent_start_char: int, ent_end_char: int
    ) -> Tuple[str, int, int]:
        """
        Trim the entity boundaries.

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
            A new text, entity start character and entity end character.
        """
        entity = text[ent_start_char:ent_end_char]

        ent_start_char += len(entity) - len(entity.lstrip())
        ent_end_char -= len(entity) - len(entity.rstrip())

        return text, ent_start_char, ent_end_char

    @staticmethod
    def _fill_ent_placeholder(
        text: str, ent_start_char: int, ent_end_char: int, placeholder: str
    ) -> Tuple[str, int, int]:
        """
        Fill the entity placeholder.

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
            A new text, entity start character and entity end character.
        """
        text = text[0:ent_start_char] + placeholder + text[ent_end_char:]
        ent_end_char = ent_start_char + len(placeholder)

        return text, ent_start_char, ent_end_char

    def _prepare_ent(self, ent: Span) -> Tuple[str, int, int]:
        """
        Prepare the entity for prediction.

        Applies the configured settings to the entity.

        Parameters
        ----------
        ent
            The entity.

        Returns
        -------
            The modified text, entity start character and entity end character.
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
            The probability.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probs = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1)

        return prob_aggregator(
            sum(pos[prob_indices]) for pos in probs[start_token : end_token + 1]
        )


@clinlp_component(
    name="clinlp_negation_transformer",
    requires=["doc.spans"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_negation_transformer,
)
class NegationTransformer(QualifierTransformer):
    """
    Transformer-based negation detector.

    Parameters
    ----------
    nlp
        The ``spaCy`` language pipeline.
    absence_threshold
        The threshold for absence. Will classify qualifier as ``Presence.Absent`` if
        ``prediction`` < ``absence_threshold``.
    presence_threshold
        The threshold for presence. Will classify qualifier as ``Presence.Present`` if
        ``prediction`` > ``presence_threshold``.
    """

    PRETRAINED_MODEL_NAME_OR_PATH = "UMCU/MedRoBERTa.nl_NegationDetection"
    REVISION = "83068ba132b6ce38e9f668c1e3ab636f79b774d3"

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH, revision=REVISION
    )
    model = RobertaForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH, revision=REVISION
    )

    def __init__(
        self,
        nlp: Language,
        absence_threshold: float = _defaults_negation_transformer["absence_threshold"],
        presence_threshold: float = _defaults_negation_transformer[
            "presence_threshold"
        ],
        **kwargs,
    ) -> None:
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
        Detect qualifiers.

        Prepares the entity, then predicts the probability of the qualifier. If the
        probability is below the absence threshold, the qualifier is classified as
        "Absent". If the probability is above the presence threshold, the qualifier is
        classified as "Present". Otherwise, it is classified as "Uncertain".

        Parameters
        ----------
        doc
            The ``Doc`` object.
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
    default_config=_defaults_experiencer_transformer,
)
class ExperiencerTransformer(QualifierTransformer):
    """
    Transformer-based experiencer detector.

    Currently, only detects ``Experiencer.Patient`` (default) and ``Experiencer.Family``
    -- ``Experiencer.Other`` is not yet implemented.

    Parameters
    ----------
    nlp
        The ``spaCy`` language pipeline.
    family_threshold
        The threshold for family. Will classify qualifier as ``Experiencer.Family`` if
        ``prediction`` > ``family_threshold``.

    """

    PRETRAINED_MODEL_NAME_OR_PATH = "UMCU/MedRoBERTa.nl_Experiencer"
    REVISION = "d9318c4b2b0ab0dfe50afedca58319b2369f1a71"

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH, revision=REVISION
    )
    model = RobertaForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH, revision=REVISION
    )

    def __init__(
        self,
        nlp: Language,
        family_threshold: float = _defaults_experiencer_transformer["family_threshold"],
        **kwargs,
    ) -> None:
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
        Detect qualifiers.

        Prepares the entity, then predicts the probability of the qualifier. If the
        probability is above the family threshold, the qualifier is classified as
        "Family". Otherwise, it is classified as "Patient" (default).

        Parameters
        ----------
        doc
            The ``Doc`` object.
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
