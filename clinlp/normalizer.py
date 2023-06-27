import unicodedata

import spacy
from spacy import Language
from spacy.tokens import Doc

_defaults_normalizer = {"lowercase": True, "map_non_ascii": True}


@spacy.language.Language.factory(
    "clinlp_normalizer",
    assigns=["token.norm"],
)
def make_normalizer(nlp: Language, name: str, **_defaults_normalizer):
    return Normalizer(**_defaults_normalizer)


class Normalizer:
    def __init__(self, lowercase=True, map_non_ascii=True):
        self.lowercase = lowercase
        self.map_non_ascii = map_non_ascii

    @staticmethod
    def _lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def _map_non_ascii_char(char: str) -> str:
        """
        TODO: What if there's a non-mappable ascii (µ) and a mappable (é) in the same string?
        """

        s = unicodedata.normalize("NFD", char)
        s = str(s.encode("ascii", "ignore").decode("utf-8"))

        return s if len(s) > 0 else char

    def _map_non_ascii_string(self, text: str) -> str:
        return "".join(self._map_non_ascii_char(char) for char in text)

    def __call__(self, doc: Doc):
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
