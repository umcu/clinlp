import spacy

from clinlp.tokenize import create_tokenizer


def create_model():
    nlp = spacy.blank("nl")
    nlp.tokenizer = create_tokenizer(nlp)

    return nlp
