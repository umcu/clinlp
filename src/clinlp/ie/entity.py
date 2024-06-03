import intervaltree as ivt
import numpy as np
import pandas as pd
import pydantic
from spacy.language import Doc, Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import Pipe
from spacy.tokens import Span

from clinlp.ie.term import Term, _defaults_term
from clinlp.util import clinlp_component

SPANS_KEY = "ents"

_defaults_entity_matcher = {
    "resolve_overlap": False,
}


_non_phrase_matcher_fields = ["proximity", "fuzzy", "fuzzy_min_len"]


def create_concept_dict(path: str, concept_col: str = "concept") -> dict:
    """Transforms source concept data to a dictionary that the clinlp
    entity matcher can read. Takes the path to a csv file where each
    row is a distinct word or sentence (term) that belongs to a concept.
    """

    df = pd.read_csv(path).replace([np.nan], [None])

    try:
        df["term"] = df.apply(lambda x: Term(**x.to_dict()), axis=1)
    except pydantic.ValidationError as e:
        msg = (
            "There is a value in your input csv which cannot be"
            "parsed. Please refer to the above error for more details."
        )

        raise RuntimeError(msg) from e

    df = df.groupby(concept_col)["term"].apply(list).reset_index()

    return dict(zip(df["concept"], df["term"]))


@clinlp_component(name="clinlp_entity_matcher")
class DeprecatedEntityMatcher(Pipe):
    def __init__(self) -> None:
        msg = (
            "The clinlp_entity_matcher has been renamed "
            "clinlp_rule_based_entity_matcher."
        )

        raise RuntimeError(msg)


@clinlp_component(
    name="clinlp_rule_based_entity_matcher",
    assigns=["doc.spans"],
    default_config=_defaults_term | _defaults_entity_matcher,
)
class RuleBasedEntityMatcher(Pipe):
    def __init__(
        self,
        nlp: Language,
        attr: str = _defaults_term["attr"],
        proximity: int = _defaults_term["proximity"],
        fuzzy: int = _defaults_term["fuzzy"],
        fuzzy_min_len: int = _defaults_term["fuzzy_min_len"],
        pseudo: bool = _defaults_term["pseudo"],  # noqa: FBT001
        resolve_overlap: bool = _defaults_entity_matcher["resolve_overlap"],  # noqa: FBT001
    ) -> None:
        self.nlp = nlp
        self.attr = attr

        self.resolve_overlap = resolve_overlap

        self.term_args = {
            "attr": attr,
            "proximity": proximity,
            "fuzzy": fuzzy,
            "fuzzy_min_len": fuzzy_min_len,
            "pseudo": pseudo,
        }

        self._matcher = Matcher(self.nlp.vocab)
        self._phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=self.attr)

        self._terms = {}
        self._concepts = {}

    @property
    def _use_phrase_matcher(self) -> bool:
        return all(
            self.term_args[field] == _defaults_term[field]
            for field in _non_phrase_matcher_fields
            if field in self.term_args
        )

    def load_concepts(self, concepts: dict) -> None:
        for concept, concept_terms in concepts.items():
            for concept_term in concept_terms:
                identifier = str(len(self._terms))

                self._terms[identifier] = concept_term
                self._concepts[identifier] = concept

                if isinstance(concept_term, str):
                    if self._use_phrase_matcher:
                        self._phrase_matcher.add(
                            key=identifier, docs=[self.nlp(concept_term)]
                        )
                    else:
                        concept_term = Term(
                            phrase=concept_term, **self.term_args
                        ).to_spacy_pattern(self.nlp)
                        self._matcher.add(key=identifier, patterns=[concept_term])

                elif isinstance(concept_term, list):
                    self._matcher.add(key=identifier, patterns=[concept_term])

                elif isinstance(concept_term, Term):
                    term_args_with_override = {}

                    for field, value in self.term_args.items():
                        if getattr(concept_term, field) is not None:
                            term_args_with_override[field] = getattr(
                                concept_term, field
                            )
                        else:
                            term_args_with_override[field] = value

                    self._matcher.add(
                        key=identifier,
                        patterns=[
                            Term(
                                phrase=concept_term.phrase, **term_args_with_override
                            ).to_spacy_pattern(self.nlp)
                        ],
                    )

                else:
                    msg = (
                        f"Not sure how to load a term with type {type(concept_term)}, "
                        f"please provide str, list or clinlp.Term"
                    )
                    raise TypeError(msg)

    def _get_matches(self, doc: Doc) -> list[tuple[int, int, int]]:
        if len(self._terms) == 0:
            msg = "No concepts added."
            raise RuntimeError(msg)

        matches = []

        if len(self._matcher) > 0:
            matches += list(self._matcher(doc))

        if len(self._phrase_matcher) > 0:
            matches += list(self._phrase_matcher(doc))

        return matches

    @staticmethod
    def _resolve_ents_overlap(ents: list[Span]) -> list[Span]:
        """
        Resolves overlap between spans, by taking the longest.

        Args:
            ents: The input Spans, with possible overlap.

        Returns: The Spans without any overlap.
        """

        if len(ents) == 0:
            return ents

        ents = sorted(ents, key=lambda span: span.start)

        disjoint_ents = [ents[0]]

        for _, ent in enumerate(ents[1:]):
            if ent.start < disjoint_ents[-1].end:
                if len(str(disjoint_ents[-1])) < len(str(ent)):
                    disjoint_ents[-1] = ent
            else:
                disjoint_ents.append(ent)

        return disjoint_ents

    def __call__(self, doc: Doc) -> Doc:
        matches = self._get_matches(doc)

        pos_matches = []
        neg_matches = ivt.IntervalTree()

        for match_id, start, end in matches:
            rule_id = self.nlp.vocab.strings[match_id]
            term = self._terms[rule_id]

            if isinstance(term, Term) and term.pseudo:
                neg_matches[start:end] = rule_id
            else:
                pos_matches.append((rule_id, start, end))

        ents = doc.spans.get(SPANS_KEY, [])

        for rule_id, start, end in pos_matches:
            if not any(
                self._concepts[rule_id] == self._concepts[neg_match_id.data]
                for neg_match_id in neg_matches.overlap(start, end)
            ):
                ents.append(
                    Span(doc=doc, start=start, end=end, label=self._concepts[rule_id])
                )

        if self.resolve_overlap:
            ents = self._resolve_ents_overlap(ents)

        doc.spans[SPANS_KEY] = ents

        return doc
