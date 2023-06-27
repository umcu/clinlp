import statistics
from typing import Optional

import torch
from spacy import Language
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, RobertaForTokenClassification

from clinlp.qualifier.qualifier import QUALIFIERS_ATTR, Qualifier, QualifierDetector

_defaults_negation_transformer = {
    "token_window": 32,
    "threshold": 0.5,
    "placeholder": None
}

TRANSFORMER_REPO = "UMCU/MedRoBERTa.nl_NegationDetection"


@Language.factory(name="clinlp_negation_transformer", requires=["doc.ents"], assigns=[f"span._.{QUALIFIERS_ATTR}"])
def make_negation_transformer(nlp: Language, name: str, **_defaults_negation_transformer):
    return NegationTransformer(nlp, **_defaults_negation_transformer)


class NegationTransformer(QualifierDetector):
    def __init__(
        self,
        nlp: Language,
        token_window: int = _defaults_negation_transformer["token_window"],
        threshold: float = _defaults_negation_transformer["threshold"],
        placeholder: Optional[str] = _defaults_negation_transformer["placeholder"],
    ):
        self.nlp = nlp
        self.token_window = token_window
        self.threshold = threshold
        self.placeholder = placeholder

        self.negation_qualifier = Qualifier("Negation", ["Affirmed", "Negated"])

        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_REPO)
        self.model = RobertaForTokenClassification.from_pretrained(TRANSFORMER_REPO)

    def get_text_span_for_entity(self, ent: Span):
        start_token_i = max(0, ent.start - self.token_window)
        end_token_i = min(len(ent.doc), ent.end + self.token_window)

        text_span = ent.doc[start_token_i:end_token_i]

        ent_start_char = ent.start_char - text_span.start_char
        ent_end_char = ent.end_char - text_span.start_char

        return str(text_span), ent_start_char, ent_end_char

    def trim_ent(self, text: str, ent_start_char: int, ent_end_char: int):
        entity = text[ent_start_char:ent_end_char]

        ent_start_char += len(entity) - len(entity.lstrip())
        ent_end_char -= len(entity) - len(entity.rstrip())

        return text, ent_start_char, ent_end_char

    def use_placeholder(self, text: str, ent_start_char: int, ent_end_char: int):
        text = text[0:ent_start_char] + self.placeholder + text[ent_end_char:]
        ent_end_char = ent_start_char + len(self.placeholder)

        return (text, ent_start_char, ent_end_char)

    def negation_probs(self, text: str, ent_start_char: int, ent_end_char: int):
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probas = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1) + 1

        probs = [
            {
                "token": input_tokens[idx],
                "proba_negated": proba_arr[0] + proba_arr[2],
                "in_ent": True if start_token <= idx < end_token else False,
            }
            for idx, proba_arr in enumerate(probas)
        ]

        return probs, start_token, end_token

    def detect_qualifiers(self, doc: Doc):
        for ent in doc.ents:
            text, ent_start_char, ent_end_char = self.get_text_span_for_entity(ent)
            text, ent_start_char, ent_end_char = self.trim_ent(text, ent_start_char, ent_end_char)

            if self.placeholder is not None:
                text, ent_start_char, ent_end_char = self.use_placeholder(text, ent_start_char, ent_end_char)

            probs, start_token, end_token = self.negation_probs(text, ent_start_char, ent_end_char)

            prob = statistics.mean(pos["proba_negated"] for pos in probs[start_token:end_token])

            if prob > self.threshold:
                self.add_qualifier_to_ent(ent, self.negation_qualifier.Negated)

        return doc
