"""Component for sentencizing text."""

from spacy.pipeline import Pipe
from spacy.tokens import Doc, Token

from clinlp.util import clinlp_component


@clinlp_component(
    name="clinlp_sentencizer", assigns=["token.is_sent_start", "doc.sents"]
)
class Sentencizer(Pipe):
    """
    ``spaCy`` pipeline component for sentencizing text.

    Uses the following logic for detecting sentence boundaries: any character included
    in ``sent_end_chars`` can mark the end of a sentence. The actual sentence boundary
    then occurs at the next token that either:
        * Is an alphanumeric token
        * Starts with ``[``
        * Is included in ``sent_start_punct``
    """

    def __init__(
        self,
        *,
        sent_end_chars: tuple[str] = (".", "!", "?", "\n", "\r"),
        sent_start_punct: tuple[str] = ("-", "*", "[", "("),
    ) -> None:
        r"""
        Create a sentencizer.

        Parameters
        ----------
        sent_end_chars
            Any character that can end a sentence.
        sent_start_punct
            Any punctuation that is allowed to start a sentence.
        """
        self.sent_end_chars = set(sent_end_chars)
        self.sent_start_punct = set(sent_start_punct)

    def _token_can_start_sent(self, token: Token) -> bool:
        """
        Determine whether a token can start a sentence.

        Parameters
        ----------
        token
            The token to check.

        Returns
        -------
        ``bool``
            Whether the token can start a sentence.
        """
        return (
            token.text[0].isalnum()
            or (token.text[0] in {"["})
            or (token.text in self.sent_start_punct)
        )

    def _token_can_end_sent(self, token: Token) -> bool:
        """
        Determine whether a token can end a sentence.

        Parameters
        ----------
        token
            The token to check.

        Returns
        -------
        ``bool``
            Whether the token can end a sentence.
        """
        return token.text in self.sent_end_chars

    def _compute_sentence_starts(self, doc: Doc) -> list[bool]:
        """
        Compute which tokens start a sentence.

        Parameters
        ----------
        doc
            The document to sentencize.

        Returns
        -------
        ``list[bool]``
            A list of booleans indicating whether each token marks the start of a
            sentence.
        """
        if len(doc) == 0:
            return []

        sentence_starts = [False] * len(doc)

        if self._token_can_start_sent(doc[0]):
            sentence_starts[0] = True

        seen_end_char = True

        for i, token in enumerate(doc):
            if seen_end_char and self._token_can_start_sent(token):
                sentence_starts[i] = True
                seen_end_char = False

            if self._token_can_end_sent(token):
                seen_end_char = True

        return sentence_starts

    def __call__(self, doc: Doc) -> Doc:
        """
        Sentencize the text in the document.

        Parameters
        ----------
        doc
            The document containing the text to sentencize.

        Returns
        -------
        ``Doc``
            The document, with ``token.is_sent_start`` set to whether each token
            starts a sentence.
        """
        if len(doc) == 0:
            return doc

        sentence_starts = self._compute_sentence_starts(doc)

        for is_sent_start, token in zip(sentence_starts, doc, strict=True):
            token.is_sent_start = is_sent_start

        return doc
