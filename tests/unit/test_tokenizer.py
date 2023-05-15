import spacy

from clinlp.tokenizer import make_tokenizer


class TestUnitTokenizer:
    def test_create_tokenizer(self):
        _ = make_tokenizer(spacy.blank("nl"))
