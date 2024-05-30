from clinlp.ie import SPANS_KEY


class TestNormalizerIntegration:
    def test_normalizer_with_entity_matching(self, nlp):
        # Arrange
        nlp.add_pipe("clinlp_normalizer")

        ruler = nlp.add_pipe(
            "span_ruler",
            config={"phrase_matcher_attr": "NORM", "spans_key": SPANS_KEY},
        )

        concepts = {"symptomen": ["caries"]}

        for concept, terms in concepts.items():
            ruler.add_patterns([{"label": concept, "pattern": term} for term in terms])

        # Act
        doc = nlp("patient heeft veel last van cariës")

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert doc.spans[SPANS_KEY][0].text == "cariës"
