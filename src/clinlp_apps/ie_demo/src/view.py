"""Views for the Information Extraction demo."""

import re

import dash_bootstrap_components as dbc
from dash import dcc, html

import clinlp
from clinlp_apps.ie_demo.src.utils import RESOURCE_PATH

TEXT_AREA_WIDTH = 1000
TEXT_AREA_HEIGHT = 300
LINE_HEIGHT = "2.5"

TOKEN_STYLE = {
    "background-color": "#f0f0f0",
    "padding": "6px",
    "margin": "2px",
    "border-radius": "8px",
}

SENTENCE_MARKER_SYMBOL = "»"
NEWLINE_SYMBOL = "⏎"
WHITESPACE_SYMBOL = "⌴"

SAMPLE_TEXT_FILE = RESOURCE_PATH / "sample_text.txt"

with SAMPLE_TEXT_FILE.open() as file:
    SAMPLE_TEXT = file.read()


text_area = html.Div(
    [
        dcc.Textarea(
            id="text-input",
            value=SAMPLE_TEXT,
            style={"width": "100%", "height": TEXT_AREA_HEIGHT},
        ),
    ]
)

content = [
    dcc.Store(id="doc"),
    html.H1("Clinlp Information Extraction Demo"),
    html.Span(f"clinlp v{clinlp.__version__}"),
    html.Br(),
    html.Br(),
    text_area,
    html.Br(),
    dcc.Tabs(
        id="output-tabs",
        value="qualifiers",
        children=[
            dcc.Tab(label="Qualifiers", value="qualifiers"),
            dcc.Tab(label="Entities", value="entities"),
            dcc.Tab(label="Sentences", value="sentences"),
            dcc.Tab(label="Normalizer", value="normalizer"),
            dcc.Tab(label="Tokens", value="tokens"),
        ],
        style={"margin-bottom": "20px", "border": "1px solid #f0f0f0"},
    ),
    html.P(id="output-text-area", style={"line-height": LINE_HEIGHT}),
]

layout = dbc.Container(
    content,
    style={"padding": "20px", "margin-top": "20px", "width": TEXT_AREA_WIDTH},
)

sentence_marker = html.Span(
    SENTENCE_MARKER_SYMBOL,
    style={"font-weight": "bold", "color": "grey", "margin": "8px"},
)

whitespace = html.Span(" ")


def format_text(text: str) -> list[html.Span]:
    """
    Wrap raw text in html.Span elements. Additionally detects linebreaks.

    Parameters
    ----------
    text
        The input text.

    Returns
    -------
        The formatted text.
    """
    output = []

    lines = re.split(r"\n|\r", text)

    for i, line in enumerate(lines):
        output.append(line)

        if i < len(lines) - 1:
            output.append(html.Br())

    return output


def format_token(text: str) -> html.Span:
    """
    Format a token as an html.Span element.

    Parameters
    ----------
    text
        The input text.

    Returns
    -------
        The formatted token.
    """
    if text in ("\n", "\r"):
        text = NEWLINE_SYMBOL

    if text in (" "):
        text = WHITESPACE_SYMBOL

    return html.Span(
        text,
        style=TOKEN_STYLE,
    )


def format_entity(
    text: str,
    sub_label: str | None = None,
    sup_label: str | None = None,
    bg_color: str = "#e6f7ff",
    label_text_color: str = "grey",
) -> html.Span:
    """
    Format an entity as an html.Span element.

    Can place two labels, right above and right below the entity text.

    Parameters
    ----------
    text
        The entity text
    sub_label, optional
        The label to place right below the text.
    sup_label, optional
        The label to place right above the text.
    bg_color, optional
        The background color.
    label_color, optional
        The label text color.

    Returns
    -------
        The formatted entity.
    """
    entity_style = {
        "background-color": bg_color,
        "padding": "6px",
        "margin": "2px",
        "border-radius": "8px",
    }

    entity_text_style = {"font-weight": "bold"}

    label_style = {
        "color": label_text_color,
        "position": "relative",
        "display": "block",
        "font-size": "0.8em",
        "line-height": 1.2,
    }

    entity = [
        html.Span(text, style=entity_text_style),
    ]

    if sup_label or sub_label:
        label_spans = []

        if sup_label is not None:
            label_spans.append(
                html.Sup(
                    sup_label,
                    style=label_style,
                )
            )

        if sub_label is not None:
            label_spans.append(
                html.Sub(
                    sub_label,
                    style=label_style,
                )
            )

        entity.append(
            html.Span(
                label_spans, style={"display": "inline-block", "margin-left": "4px"}
            )
        )

    return html.Span(
        entity,
        style=entity_style,
    )
