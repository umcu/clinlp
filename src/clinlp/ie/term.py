"""``Term`` class, which is used for rule based entity matching."""

from typing import Optional

import pydantic
from spacy.language import Language

from clinlp.util import get_class_init_signature


class Term(pydantic.BaseModel):
    """A single term used for rule based entity matching."""

    phrase: str
    """The literal phrase to match."""

    attr: Optional[str] = "TEXT"
    """The attribute to match on."""

    proximity: Optional[int] = 0
    """ The number of tokens to allow between each token in the phrase."""

    fuzzy: Optional[int] = 0
    """The threshold for fuzzy matching."""

    fuzzy_min_len: Optional[int] = 0
    """The minimum length for fuzzy matching."""

    pseudo: Optional[bool] = False
    """Whether this term is a pseudo-term, which is excluded from matches."""

    model_config = {"extra": "ignore"}

    # ensures Term is accepted as positional argument for readability
    def __init__(self, phrase: str, **kwargs) -> None:
        super().__init__(phrase=phrase, **kwargs)

    @classmethod
    def defaults(cls) -> dict:
        """
        Get the default values for each term attribute, if any.

        Returns
        -------
        ``dict``
            The default values for each attribute, if any.
        """
        _, defaults = get_class_init_signature(cls)

        return defaults

    @property
    def fields_set(self) -> set[str]:
        """
        Get the fields set for this term.

        Returns
        -------
        ``set[str]``
            The fields set for this term.
        """
        return self.__pydantic_fields_set__

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
        spacy_pattern = []

        phrase_tokens = [token.text for token in nlp.tokenizer(self.phrase)]

        for i, token in enumerate(phrase_tokens):
            if (self.fuzzy > 0) and (len(token) >= self.fuzzy_min_len):
                token_pattern = {f"FUZZY{self.fuzzy}": token}
            else:
                token_pattern = token

            spacy_pattern.append({self.attr: token_pattern})

            if i != len(phrase_tokens) - 1:
                spacy_pattern += [{"OP": "?"}] * self.proximity

        return spacy_pattern
