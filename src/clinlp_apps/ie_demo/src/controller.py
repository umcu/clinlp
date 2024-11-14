"""Controller for the Information Extraction demo."""

import pickle

import spacy
from dash import Input, Output, State, callback, html

from clinlp_apps.ie_demo.src.model import check_overlapping_entities, get_model
from clinlp_apps.ie_demo.src.utils import simple_label
from clinlp_apps.ie_demo.src.view import (
    format_entity,
    format_text,
    format_token,
    sentence_marker,
    whitespace,
)

OVERLAP_ERROR = "⚠️ Cannot visualize overlapping entities, please modify input."
NON_QUALIFIED_BGCOLOR = "#e6ffe6"
QUALIFIED_BGCOLOR = "#ffe6e6"


nlp = get_model()


def tokens(doc: spacy.language.Doc) -> list[html.Span]:
    """
    Produce visual output for the tokens tab.

    Parameters
    ----------
    doc
        The input document.

    Returns
    -------
        The visual output for the tokens tab.
    """
    output = []

    for token in doc:
        output.append(format_token(token.text))
        output.append(whitespace)

    return output


def normalized(doc: spacy.language.Doc) -> list[html.Span]:
    """
    Produce visual output for the normalizer tab.

    Parameters
    ----------
    doc
        The input document.

    Returns
    -------
        The visual output for the normalizer tab.
    """
    output = []

    for token in doc:
        output.append(format_token(token.norm_))
        output.append(whitespace)

    return output


def sentences(doc: spacy.language.Doc) -> list[html.Span]:
    """
    Produce visual output for the sentences tab.

    Parameters
    ----------
    doc
        The input document.

    Returns
    -------
        The visual output for the sentences tab.
    """
    output = []

    for sent in doc.sents:
        output.append(sentence_marker)
        output.extend(format_text(sent.text))

    return output


def entities(doc: spacy.language.Doc) -> list[html.Span]:
    """
    Produce visual output for the entities tab.

    Parameters
    ----------
    doc
        The input document.

    Returns
    -------
        The visual output for the entities tab.
    """
    if check_overlapping_entities(doc):
        return [html.Span(OVERLAP_ERROR)]

    output = []
    i = 0

    for ent in doc.spans["ents"]:
        output.extend(format_text(doc[i : ent.start].text))
        output.append(format_entity(ent.text, sub_label=simple_label(ent.label_)))
        i = ent.end

    output.extend(format_text(doc[i:].text))

    return output


def qualifiers(doc: spacy.language.Doc) -> list[html.Span]:
    """
    Produce visual output for the qualifiers tab.

    Parameters
    ----------
    doc
        The input document.

    Returns
    -------
        The visual output for the qualifiers tab.
    """
    if check_overlapping_entities(doc):
        return [html.Span(OVERLAP_ERROR)]

    output = []
    i = 0

    for ent in doc.spans["ents"]:
        output.extend(format_text(doc[i : ent.start].text))

        q_label = ",".join(q.value for q in ent._.qualifiers if not q.is_default)

        entity = {
            "text": ent.text,
            "sub_label": simple_label(ent.label_),
        }

        # Non-qualified
        if len(q_label) == 0:
            entity["bg_color"] = NON_QUALIFIED_BGCOLOR

        # Qualified
        else:
            entity["sup_label"] = simple_label(q_label)
            entity["bg_color"] = QUALIFIED_BGCOLOR

        output.append(format_entity(**entity))
        i = ent.end

    output.extend(format_text(doc[i:].text))

    return output


@callback(
    Output("output-text-area", "children"),
    State("doc", "data"),
    Input("output-tabs", "value"),
    Input("doc", "modified_timestamp"),
    prevent_initial_call=True,
)
def render_tab(doc_data: list[str], tab: str, _: str) -> list[html.Span]:
    """
    Render the visual output for the selected tab.

    Parameters
    ----------
    doc_data
        The serialized doc object.
    tab
        The name of the selected tab.

    Returns
    -------
        The visual output for the selected tab.
    """
    doc = pickle.loads(doc_data.encode("latin1"))  # noqa S301

    renderers = {
        "qualifiers": qualifiers,
        "entities": entities,
        "sentences": sentences,
        "normalizer": normalized,
        "tokens": tokens,
    }

    return renderers[tab](doc)


@callback(
    Output("doc", "data"),
    Input("text-input", "value"),
)
def process_text(text: str) -> list[str]:
    """
    Process the changed input and store doc object.

    Parameters
    ----------
    text
        The input text.

    Returns
    -------
        A serialized doc object for storage.
    """
    doc = nlp(text)

    return pickle.dumps(doc).decode("latin1")
