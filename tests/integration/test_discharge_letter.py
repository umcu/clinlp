import re
from html import escape

from spacy import displacy

from clinlp.ie import SPANS_KEY, Term

DISCHARGE_LETTER = """ONTSLAGBRIEF CARDIOLOGIE — SYNTHETISCH VOORBEELD
Alle gegevens in deze brief zijn fictief.

Patiënt: Henk de Vries

Reden van opname
Patiënt werd opgenomen wegens palpitaties sinds twee dagen.
Er was geen sprake van pijn op de borst.

Voorgeschiedenis en familieanamnese
Bij patiënt is atriumfibrilleren bekend.
Zijn moeder heeft diabetes mellitus type 2 waarvoor zij wordt behandeld.

Beloop en behandeling
Tijdens de opname werd een elektrische cardioversie uitgevoerd.
Hierna herstelde het sinusritme en voelde patiënt zich beter.

Ontslagmedicatie
metoprolol 50 mg, eenmaal daags oraal.

Controle en advies
Poliklinische controle bij de cardioloog over vier weken.
Bij terugkeer van klachten contact opnemen met de behandelaar.
"""

TERMS = {
    "Medicatie": [Term("metoprolol")],
    "Symptoom": [Term("palpitaties"), Term("pijn op de borst")],
    "Ziekte": [Term("atriumfibrilleren"), Term("diabetes mellitus type 2")],
    "Procedure": [Term("elektrische cardioversie")],
    "Persoonsgegeven": [Term("Henk de Vries")],
}


def _render_qualifier_report(doc):
    entities = sorted(doc.spans[SPANS_KEY], key=lambda ent: ent.start_char)
    tooltips = []
    for ent in entities:
        lines = [f"{ent.text} — {ent.label_}"]
        for qualifier in sorted(ent._.qualifiers, key=lambda item: item.name):
            probability = (
                f"{qualifier.prob:.3f}"
                if qualifier.prob is not None
                else "niet beschikbaar"
            )
            lines.append(
                f"{qualifier.name}: {qualifier.value} | kans: {probability}"
                f" | standaard: {'ja' if qualifier.is_default else 'nee'}"
            )
        tooltips.append("\n".join(lines))

    markup = displacy.render(
        {
            "text": doc.text,
            "ents": [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in entities
            ],
            "title": None,
        },
        style="ent",
        manual=True,
        page=False,
        jupyter=False,
        options={
            "colors": {
                "Medicatie": "#d1e8ff",
                "Symptoom": "#ffe1b3",
                "Ziekte": "#ffd4dc",
                "Procedure": "#d5efcf",
                "Persoonsgegeven": "#e6d9ff",
            }
        },
    )

    # displaCy has no qualifier tooltip option; enrich its entity marks only.
    tooltip_iterator = iter(tooltips)

    def add_tooltip(match):
        tooltip = escape(next(tooltip_iterator), quote=True)
        return f'{match.group()} tabindex="0" title="{tooltip}" aria-label="{tooltip}"'

    markup = re.sub(r"<mark\b", add_tooltip, markup)
    return f"""<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetische ontslagbrief — clinlp qualifiers</title>
<style>
body {{ margin: 0; background: #f3f5f8; color: #202938;
        font-family: system-ui, sans-serif; }}
main {{ max-width: 960px; margin: 2rem auto; padding: 2rem;
        background: white; border-radius: 12px; box-shadow: 0 4px 24px #0001; }}
h1 {{ font-size: 1.6rem; }}
p {{ color: #536174; line-height: 1.6; }}
.entity {{ cursor: help; }}
.entity:hover, .entity:focus {{ outline: 2px solid #425b87; outline-offset: 3px; }}
</style>
</head>
<body><main>
<h1>Ontslagbrief met qualifiers</h1>
<p>Uitsluitend synthetische gegevens. Beweeg de muis over een gemarkeerde term
voor Presence, Experiencer, PII en MED_NER, inclusief kansen en standaardwaarden.
De gekleurde labels zijn de categorieën uit de woordenlijst, niet de
modelvoorspellingen.
Modeluitvoer is geen klinisch advies.</p>
{markup}
</main></body>
</html>"""


def test_discharge_letter_all_transformer_qualifiers(nlp, tmp_path):
    # Arrange
    nlp.add_pipe("clinlp_normalizer")
    nlp.add_pipe("clinlp_sentencizer")
    matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher")
    matcher.add_terms_from_dict(TERMS)

    for component in (
        "clinlp_experiencer_transformer",
        "clinlp_negation_transformer",
        "clinlp_pii_transformer",
        "clinlp_cardioner_transformer",
    ):
        nlp.add_pipe(component, config={"token_window": 32, "placeholder": None})

    # Act
    doc = nlp(DISCHARGE_LETTER)
    report = _render_qualifier_report(doc)
    report_path = tmp_path / "discharge_letter_qualifiers.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nQualifier report: {report_path.resolve()}")  # noqa: T201

    # Assert
    entities = doc.spans[SPANS_KEY]
    expected_entities = {
        (term.phrase, category)
        for category, terms in TERMS.items()
        for term in terms
    }
    assert len(entities) == len(expected_entities)
    assert {(ent.text, ent.label_) for ent in entities} == expected_entities
    for ent in entities:
        qualifiers = ent._.qualifiers
        assert len(qualifiers) == 4
        assert {qualifier.name for qualifier in qualifiers} == {
            "Presence", "Experiencer", "PII", "MED_NER"
        }
        for qualifier in qualifiers:
            assert qualifier.prob is None or 0 <= qualifier.prob <= 1
        for component in nlp.pipeline:
            detector = component[1]
            if hasattr(detector, "qualifier_classes"):
                for qualifier_class in detector.qualifier_classes.values():
                    qualifier = next(
                        q for q in qualifiers if q.name == qualifier_class.name
                    )
                    assert qualifier.value in qualifier_class.values

    assert report_path.is_file()
    assert report.count('tabindex="0"') == len(entities)
    assert report.count('aria-label="') == len(entities)
    assert "Presence:" in report
    assert "Experiencer:" in report
    assert "PII:" in report
    assert "MED_NER:" in report
