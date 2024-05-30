from typing import Optional

from spacy.pipeline import Pipe
from spacy.tokens import Doc, Token

from clinlp.util import clinlp_component

_defaults_sentencizer = {
    "sent_end_chars": [".", "!", "?", "\n", "\r"],
    "sent_start_punct": ["-", "*", "[", "("],
}


@clinlp_component(
    name="clinlp_sentencizer",
    assigns=["token.is_sent_start", "doc.sents"],
    default_config=_defaults_sentencizer,
)
class Sentencizer(Pipe):
    def __init__(
        self,
        sent_end_chars: Optional[list[str]] = None,
        sent_start_punct: Optional[list[str]] = None,
    ) -> None:
        self.sent_end_chars = set(
            _defaults_sentencizer["sent_end_chars"]
            if sent_end_chars is None
            else sent_end_chars
        )
        self.sent_start_punct = set(
            _defaults_sentencizer["sent_start_punct"]
            if sent_start_punct is None
            else sent_start_punct
        )

    def _token_can_start_sent(self, token: Token) -> bool:
        """
        Determines whether a token can start a sentence
        """
        return (
            token.text[0].isalnum()
            or (token.text[0] in {"["})
            or (token.text in self.sent_start_punct)
        )

    def _token_can_end_sent(self, token: Token) -> bool:
        """
        Determines whether a token can end a sentence
        """
        return token.text in self.sent_end_chars

    def _get_sentence_starts(self, doc: Doc) -> list[bool]:
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
        if len(doc) == 0:
            return doc

        sentence_starts = self._get_sentence_starts(doc)

        for is_sent_start, token in zip(sentence_starts, doc):
            token.is_sent_start = is_sent_start

        return doc
