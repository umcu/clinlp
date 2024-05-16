import statistics
from typing import Callable, Optional, Tuple

import torch
from spacy import Language
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, RobertaForTokenClassification

from clinlp.ie.qualifier.qualifier import (
    ATTR_QUALIFIERS,
    QualifierClass,
    QualifierDetector,
)
from clinlp.util import clinlp_autocomponent

HF_FROM_PRETRAINED_NEGATION = {
    "pretrained_model_name_or_path": "UMCU/MedRoBERTa.nl_NegationDetection",
    "revision": "83068ba132b6ce38e9f668c1e3ab636f79b774d3",
}

HF_FROM_PRETRAINED_EXPERIENCER = {
    "pretrained_model_name_or_path": "UMCU/MedRoBERTa.nl_Experiencer",
    "revision": "d9318c4b2b0ab0dfe50afedca58319b2369f1a71",
}

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
    "other_threshold": 0.5,
}


class QualifierTransformer(QualifierDetector):
    def __init__(
        self,
        token_window: int = _defaults_qualifier_transformer["token_window"],
        strip_entities: bool = _defaults_qualifier_transformer["strip_entities"],
        placeholder: Optional[str] = _defaults_qualifier_transformer["placeholder"],
        prob_aggregator: int = _defaults_qualifier_transformer["prob_aggregator"],
    ):
        self.token_window = token_window
        self.strip_entities = strip_entities
        self.placeholder = placeholder
        self.prob_aggregator = prob_aggregator

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
        entity = text[ent_start_char:ent_end_char]

        ent_start_char += len(entity) - len(entity.lstrip())
        ent_end_char -= len(entity) - len(entity.rstrip())

        return text, ent_start_char, ent_end_char

    @staticmethod
    def _fill_ent_placeholder(
        text: str, ent_start_char: int, ent_end_char: int, placeholder: str
    ) -> Tuple[str, int, int]:
        text = text[0:ent_start_char] + placeholder + text[ent_end_char:]
        ent_end_char = ent_start_char + len(placeholder)

        return text, ent_start_char, ent_end_char

    def _prepare_ent(self, ent: Span) -> Tuple[str, int, int]:
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
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probs = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1)

        return prob_aggregator(
            sum(pos[prob_indices]) for pos in probs[start_token : end_token + 1]
        )


@Language.factory(
    name="clinlp_negation_transformer",
    requires=["doc.ents"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_negation_transformer,
)
@clinlp_autocomponent
class NegationTransformer(QualifierTransformer):
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

        self.tokenizer = AutoTokenizer.from_pretrained(**HF_FROM_PRETRAINED_NEGATION)
        self.model = RobertaForTokenClassification.from_pretrained(
            **HF_FROM_PRETRAINED_NEGATION
        )

        super().__init__(**kwargs)

    @property
    def qualifier_classes(self) -> dict[str, QualifierClass]:
        return {
            "Presence": QualifierClass(
                "Presence", ["Absent", "Uncertain", "Present"], default="Present"
            )
        }

    def _detect_qualifiers(self, doc: Doc):
        for ent in doc.ents:
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


@Language.factory(
    name="clinlp_experiencer_transformer",
    requires=["doc.ents"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_experiencer_transformer,
)
@clinlp_autocomponent
class ExperiencerTransformer(QualifierTransformer):
    def __init__(
        self,
        nlp: Language,
        other_threshold: float = _defaults_experiencer_transformer["other_threshold"],
        **kwargs,
    ) -> None:
        self.nlp = nlp
        self.other_threshold = other_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(**HF_FROM_PRETRAINED_EXPERIENCER)
        self.model = RobertaForTokenClassification.from_pretrained(
            **HF_FROM_PRETRAINED_EXPERIENCER
        )

        super().__init__(**kwargs)

    @property
    def qualifier_classes(self) -> dict[str, QualifierClass]:
        return {
            "Experiencer": QualifierClass(
                "Experiencer", ["Patient", "Family"], default="Patient"
            )
        }

    def _detect_qualifiers(self, doc: Doc):
        for ent in doc.ents:
            text, ent_start_char, ent_end_char = self._prepare_ent(ent)

            prob = self._predict(
                text,
                ent_start_char,
                ent_end_char,
                prob_indices=[1, 3],
                prob_aggregator=self.prob_aggregator,
            )

            if prob > self.other_threshold:
                self.add_qualifier_to_ent(
                    ent,
                    self.qualifier_classes["Experiencer"].create("Family", prob=prob),
                )
