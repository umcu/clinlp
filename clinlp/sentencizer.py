from typing import Optional

import spacy.tokens
from spacy.language import Language

from clinlp.util import clinlp_autocomponent

_defaults_sentencizer = {"sent_end_chars": [".", "!", "?", "\n", "\r"], "sent_start_punct": ["-", "*", "[", "("]}


@Language.factory(
    "clinlp_sentencizer", assigns=["token.is_sent_start", "doc.sents"], default_config=_defaults_sentencizer
)
@clinlp_autocomponent
class Sentencizer:
    def __init__(
        self,
        sent_end_chars: Optional[list[str]] = None,
        sent_start_punct: Optional[list[str]] = None,
    ):
        self.sent_end_chars = _defaults_sentencizer["sent_end_chars"] if sent_end_chars is None else sent_end_chars
        self.sent_start_punct = (
            _defaults_sentencizer["sent_start_punct"] if sent_start_punct is None else sent_start_punct
        )

        self.sent_end_chars = set(self.sent_end_chars)
        self.sent_start_punct = set(self.sent_start_punct)

    def _token_can_start_sent(self, token: spacy.tokens.Token) -> bool:
        """
        Determines whether a token can start a sentence
        """
        return token.text[0].isalnum() or (token.text[0] in {"["}) or (token.text in self.sent_start_punct)

    def _token_can_end_sent(self, token: spacy.tokens.Token):
        """
        Determines whether a token can end a sentence
        """
        return token.text in self.sent_end_chars

    def _get_sentence_starts(self, doc: spacy.tokens.Doc) -> list[bool]:
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

    def __call__(self, doc: spacy.tokens.Doc):
        if len(doc) == 0:
            return doc

        sentence_starts = self._get_sentence_starts(doc)

        for is_sent_start, token in zip(sentence_starts, doc):
            token.is_sent_start = is_sent_start

        return doc
