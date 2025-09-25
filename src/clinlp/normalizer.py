"""Component for normalizing text."""

import unicodedata

from spacy.pipeline import Pipe
from spacy.tokens import Doc

from clinlp.util import clinlp_component


@clinlp_component(
    name="clinlp_normalizer",
    assigns=["token.norm"],
)
class Normalizer(Pipe):
    """``spaCy`` pipeline component for normalizing text."""

    def __init__(
        self,
        *,
        lowercase: bool = True,
        map_non_ascii: bool = True,
    ) -> None:
        """
        Create a normalizer.

        Parameters
        ----------
        lowercase
            Whether to lowercase text.
        map_non_ascii
            Whether to map non ascii characters to ascii counterparts.
        """
        self.lowercase = lowercase
        self.map_non_ascii = map_non_ascii

    @staticmethod
    def _lowercase(text: str) -> str:
        """
        Lowercase text.

        Parameters
        ----------
        text
            The text to lowercase.

        Returns
        -------
        ``str``
            The lowercased text.
        """
        return text.casefold()

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
        ``str``
            The mapped character. If the character is not non-ascii, it is returned as
            is.

        Raises
        ------
        ValueError
            If the input character is not of length 1.
        """
        if len(char) != 1:
            msg = (
                "Please only use the _map_non_ascii_char method on strings of length 1."
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
        ``str``
            The text with non-ascii characters mapped to their ascii counterparts.
        """
        return "".join(self._map_non_ascii_char(char) for char in text)

    def normalize(self, text: str) -> str:
        """
        Normalize text, based on the component's settings.

        Parameters
        ----------
        text
            The text to normalize.

        Returns
        -------
        ``str``
            The normalized text.
        """
        if self.lowercase:
            text = self._lowercase(text)

        if self.map_non_ascii:
            text = self._map_non_ascii_string(text)

        return text

    def __call__(self, doc: Doc) -> Doc:
        """
        Normalize text in a document.

        Parameters
        ----------
        doc
            The document containing the text to normalize.

        Returns
        -------
        ``Doc``
            The document, with ``token.norm_`` set to the normalized text.
        """
        if len(doc) == 0:
            return doc

        for token in doc:
            token.norm_ = self.normalize(token.text)

        return doc
