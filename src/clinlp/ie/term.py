"""``Term`` class, which is used for rule based entity matching."""

from typing import Optional

import pydantic
from spacy.language import Language

_defaults_term = {
    "attr": "TEXT",
    "proximity": 0,
    "fuzzy": 0,
    "fuzzy_min_len": 0,
    "pseudo": False,
}


class Term(pydantic.BaseModel):
    """A single term used for rule based entity matching."""

    phrase: str
    """The literal phrase to match."""

    attr: Optional[str] = None
    """The attribute to match on."""

    proximity: Optional[int] = None
    """ The number of tokens to allow between each token in the phrase."""

    fuzzy: Optional[int] = None
    """The threshold for fuzzy matching."""

    fuzzy_min_len: Optional[int] = None
    """The minimum length for fuzzy matching."""

    pseudo: Optional[bool] = None
    """Whether this term is a pseudo-term, which is excluded from matches."""

    model_config = {"extra": "ignore"}

    # ensures Term is accepted as positional argument for readability
    def __init__(self, phrase: str, **kwargs) -> None:
        super().__init__(phrase=phrase, **kwargs)

    def to_spacy_pattern(self, nlp: Language) -> list[dict]:
        """
        Convert the term to a ``spaCy`` pattern.

        Parameters
        ----------
        nlp
            The ``spaCy`` language model. This is used for tokenizing patterns.

        Returns
        -------
        ``list[dict]``
            The ``spaCy`` pattern.
        """
        fields = {
            field: getattr(self, field) or _defaults_term[field]
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
                    spacy_pattern.append({"OP": "?"})  # noqa: PERF401

        return spacy_pattern
