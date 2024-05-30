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
    def __init__(
        self,
        lowercase: bool = _defaults_normalizer["lowercase"],  # noqa FBT001
        map_non_ascii: bool = _defaults_normalizer["map_non_ascii"],  # noqa FBT001
    ) -> None:
        self.lowercase = lowercase
        self.map_non_ascii = map_non_ascii

    @staticmethod
    def _lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def _map_non_ascii_char(char: str) -> str:
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
        return "".join(self._map_non_ascii_char(char) for char in text)

    def __call__(self, doc: Doc) -> Doc:
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
