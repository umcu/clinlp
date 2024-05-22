import spacy

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY


class TestIntegrationEntity:
    def test_multiple_matchers(self):
        nlp = spacy.blank("clinlp")

        ruler = nlp.add_pipe("span_ruler", config={"spans_key": SPANS_KEY})
        ruler.add_patterns([{"label": "delier", "pattern": "delier"}])

        rbem = nlp.add_pipe("clinlp_rule_based_entity_matcher")
        rbem.load_concepts({"diabetes": ["diabetes"]})

        doc = nlp("De patient heeft diabetes en delier.")

        assert len(doc.ents) == 0
        assert len(doc.spans[SPANS_KEY]) == 2
        assert doc.spans[SPANS_KEY][0].label_ == "delier"
        assert doc.spans[SPANS_KEY][1].label_ == "diabetes"