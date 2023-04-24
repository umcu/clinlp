import spacy

from clinlp.sentencizer import make_sentencizer
from clinlp.tokenizer import make_tokenizer

_SPACY_LANG_CODE = "nl"


def create_model():
    nlp = spacy.blank(_SPACY_LANG_CODE)
    nlp.tokenizer = make_tokenizer(nlp)
    nlp.add_pipe("clinlp_sentencizer")

    return nlp
