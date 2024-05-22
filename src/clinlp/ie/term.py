from typing import Optional

import pydantic
from spacy.language import Language

from clinlp.ie.entity import _defaults_clinlp_ner


class Term(pydantic.BaseModel):
    phrase: str
    attr: Optional[str] = None
    proximity: Optional[int] = None
    fuzzy: Optional[int] = None
    fuzzy_min_len: Optional[int] = None
    pseudo: Optional[bool] = None

    model_config = {"extra": "ignore"}

    def __init__(self, phrase: str, **kwargs):
        """This init makes sure Term accepts phrase as a positional argument,
        which is more readable in large concept lists."""
        super().__init__(phrase=phrase, **kwargs)

    def to_spacy_pattern(self, nlp: Language):
        fields = {
            field: getattr(self, field) or _defaults_clinlp_ner[field]
            for field in ["attr", "proximity", "fuzzy", "fuzzy_min_len", "pseudo"]
        }

        spacy_pattern = []

        phrase_tokens = [token.text for token in nlp.tokenizer(self.phrase)]

        for i, token in enumerate(phrase_tokens):
            if (fields["fuzzy"] > 0) and (len(token) >= fields["fuzzy_min_len"]):
                token_pattern = {f"FUZZY{fields['fuzzy']}": token}
            else:
                token_pattern = token

            spacy_pattern.append({fields["attr"]: token_pattern})

            if i != len(phrase_tokens) - 1:
                for _ in range(fields["proximity"]):
                    spacy_pattern.append({"OP": "?"})

        return spacy_pattern
