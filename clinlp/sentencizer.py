""" TODO remove hardcoded stuff and make configurable """

from typing import Optional

import spacy.tokens


@spacy.language.Language.factory(
    "clinlp_sentencizer",
    assigns=["token.is_sent_start", "doc.sents"],
    default_config={"sent_end_chars": None, "sent_start_punct": None},
)
def make_sentencizer(
    nlp: spacy.language.Language,
    name: str,
    sent_end_chars: Optional[list[str]] = None,
    sent_start_punct: Optional[list[str]] = None,
):
    return ClinlpSentencizer(name, sent_end_chars=None, sent_start_punct=None)


class ClinlpSentencizer(spacy.pipeline.Pipe):
    default_sent_end_chars = [".", "!", "?", "\n", "\r"]
    default_sent_start_punct = ["-", "*", "[", "("]

    def __init__(
        self,
        name="clinlp_sentencizer",
        *,
        sent_end_chars: Optional[list[str]] = None,
        sent_start_punct: Optional[list[str]] = None
    ):
        self.name = name

        if sent_end_chars is None:
            self.sent_end_chars = set(self.default_sent_end_chars)
        else:
            self.sent_end_chars = set(sent_end_chars)

        if sent_start_punct is None:
            self.sent_start_punct = set(self.default_sent_start_punct)
        else:
            self.sent_start_punct = set(sent_start_punct)

    def token_can_start_sent(self, token: spacy.tokens.Token) -> bool:
        return token.text[0].isalnum() or (token.text[0] in {"["}) or (token.text in self.sent_start_punct)

    def token_can_end_sent(self, token: spacy.tokens.Token):
        return token.text in self.sent_end_chars

    def predict(self, doc: spacy.tokens.Doc) -> list[bool]:
        if len(doc) == 0:
            return []

        markings = [False] * len(doc)

        if self.token_can_start_sent(doc[0]):
            markings[0] = True

        seen_end_char = True

        for i, token in enumerate(doc):
            if seen_end_char:
                if self.token_can_start_sent(token):
                    markings[i] = True
                    seen_end_char = False

            if self.token_can_end_sent(token):
                seen_end_char = True

        return markings

    def __call__(self, doc: spacy.tokens.Doc):
        error_handler = self.get_error_handler()

        if len(doc) > 0:
            try:
                predictions = self.predict(doc)

                for prediction, token in zip(predictions, doc):
                    token.is_sent_start = prediction

            except Exception as e:
                error_handler(self.name, self, [doc], e)

        return doc
