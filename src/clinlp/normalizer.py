"""Functionality for normalizing text."""

import unicodedata

from spacy.pipeline import Pipe
from spacy.tokens import Doc

from clinlp.util import clinlp_component

_defaults_normalizer = {"lowercase": True, "map_non_ascii": True}


@clinlp_component(
    name="clinlp_normalizer",
    assigns=["token.norm"],
    default_config=_defaults_normalizer,
)
class Normalizer(Pipe):
    """
    A ``spaCy`` pipeline component for normalizing text.

    Parameters
    ----------
    lowercase
        Whether to lowercase text, by default ``True``.
    map_non_ascii
        Whether to map non ascii characters to ascii counterparts, by default ``True``.
    """

    def __init__(
        self,
        lowercase: bool = _defaults_normalizer["lowercase"],  # noqa FBT001
        map_non_ascii: bool = _defaults_normalizer["map_non_ascii"],  # noqa FBT001
    ) -> None:
        self.lowercase = lowercase
        self.map_non_ascii = map_non_ascii

    @staticmethod
    def _lowercase(text: str) -> str:
        """
        Lowercase the text.

        Parameters
        ----------
        text
            The text to lowercase.

        Returns
        -------
            The lowercased text.
        """
        return text.lower()

    @staticmethod
    def _map_non_ascii_char(char: str) -> str:
        """
        Map non-ascii characters to their ascii counterparts.

        Only handles single characters. Uses NFD normalization to decompose the
        character into its base and diacritic, then encodes the character to ascii,
        ignoring any characters that cannot be encoded. The character is then decoded
        to utf-8 and returned.

        Parameters
        ----------
        char
            The character to map.

        Returns
        -------
            The mapped character. If the character is not non-ascii, it is returned as
            is.

        Raises
        ------
        ValueError
            If the input character is not of length 1.
        """
        if len(char) != 1:
            msg = (
                "Please only use the _map_non_ascii_char method "
                "on strings of length 1."
            )
            raise ValueError(msg)

        normalized_char = unicodedata.normalize("NFD", char)
        normalized_char = str(normalized_char.encode("ascii", "ignore").decode("utf-8"))

        return normalized_char if len(normalized_char) > 0 else char

    def _map_non_ascii_string(self, text: str) -> str:
        """
        Map non-ascii characters in a string to their ascii counterparts.

        Can handle any string, rather than just a single character.

        Parameters
        ----------
        text
            The text to map non-ascii characters in.

        Returns
        -------
            The text with non-ascii characters mapped to their ascii counterparts.
        """
        return "".join(self._map_non_ascii_char(char) for char in text)

    def __call__(self, doc: Doc) -> Doc:
        """
        Normalize the text in the doc.

        Parameters
        ----------
        doc
            The doc to normalize.

        Returns
        -------
            The doc with ``token.norm_`` set to the normalized text.
        """
        if len(doc) == 0:
            return doc

        for token in doc:
            normalized_text = token.text

            if self.lowercase:
                normalized_text = self._lowercase(normalized_text)

            if self.map_non_ascii:
                normalized_text = self._map_non_ascii_string(normalized_text)

            token.norm_ = normalized_text

        return doc
