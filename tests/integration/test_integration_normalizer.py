import spacy

import clinlp  # noqa: F401


class TestNormalizerIntegration:
    def test_normalizer_with_entity_matching(self):
        nlp = spacy.blank("clinlp")
        nlp.add_pipe("clinlp_normalizer")

        ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "NORM"})

        concepts = {"symptomen": ["caries"]}

        for concept, terms in concepts.items():
            ruler.add_patterns([{"label": concept, "pattern": term} for term in terms])

        doc = nlp("patient heeft veel last van cariës")

        assert len(doc.ents) == 1
        assert doc.ents[0].text == "cariës"
