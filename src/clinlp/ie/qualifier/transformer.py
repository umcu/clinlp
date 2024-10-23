"""Transformer-based qualifier detectors."""

import statistics
from abc import abstractmethod
from collections.abc import Callable

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


class QualifierTransformer(QualifierDetector):
    """
    Transformer-based qualifier detector.

    Implements some helper methods, but cannot be used directly. Specifically, does not
    implement the abstract properties ``qualifier_classes``, ``tokenizer`` and
    ``model``, and abstract method ``_detect_qualifiers``.
    """

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

    @property
    @abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        """
        The tokenizer.

        Returns
        -------
        ``AutoTokenizer``
            The tokenizer.
        """

    @property
    @abstractmethod
    def model(self) -> RobertaForTokenClassification:
        """
        The model.

        Returns
        -------
        ``RobertaForTokenClassification``
            The model.
        """

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

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1)

        return prob_aggregator(
            sum(pos[prob_indices]) for pos in probs[start_token : end_token + 1]
        )


@clinlp_component(
    name="clinlp_negation_transformer",
    requires=["doc.spans"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
)
class NegationTransformer(QualifierTransformer):
    """Transformer-based negation detector."""

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

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH, revision=REVISION
    )
    model = RobertaForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH, revision=REVISION
    )

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
