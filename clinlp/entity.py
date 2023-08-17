from dataclasses import dataclass
from typing import Optional

import intervaltree as ivt
from spacy.language import Doc, Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span

from clinlp.util import clinlp_autocomponent

_defaults_clinlp_ner = {"attr": "TEXT", "proximity": 0, "fuzzy": 0, "fuzzy_min_len": 0, "pseudo": False}
_non_phrase_matcher_fields = ["proximity", "fuzzy", "fuzzy_min_len"]


@dataclass
class Term:
    phrase: str
    attr: Optional[str] = _defaults_clinlp_ner["attr"]
    proximity: Optional[int] = _defaults_clinlp_ner["proximity"]
    fuzzy: Optional[int] = _defaults_clinlp_ner["fuzzy"]
    fuzzy_min_len: Optional[int] = _defaults_clinlp_ner["fuzzy_min_len"]
    pseudo: Optional[bool] = _defaults_clinlp_ner["pseudo"]

    def to_spacy_pattern(self, nlp: Language):
        spacy_pattern = []

        phrase_tokens = [token.text for token in nlp.tokenizer(self.phrase)]

        for i, token in enumerate(phrase_tokens):
            if (self.fuzzy > 0) and (len(token) >= self.fuzzy_min_len):
                token_pattern = {f"FUZZY{self.fuzzy}": token}
            else:
                token_pattern = token

            spacy_pattern.append({self.attr: token_pattern})

            if i != len(phrase_tokens) - 1:
                for _ in range(self.proximity):
                    spacy_pattern.append({"OP": "?"})

        return spacy_pattern


@Language.factory(
    name="clinlp_entity_matcher",
    requires=["doc.sents", "doc.ents"],
    assigns=["doc.ents"],
    default_config=_defaults_clinlp_ner,
)
@clinlp_autocomponent
class EntityMatcher:
    def __init__(
        self,
        nlp: Language,
        attr: Optional[str] = _defaults_clinlp_ner["attr"],
        proximity: Optional[int] = _defaults_clinlp_ner["proximity"],
        fuzzy: Optional[int] = _defaults_clinlp_ner["fuzzy"],
        fuzzy_min_len: Optional[int] = _defaults_clinlp_ner["fuzzy_min_len"],
        pseudo: Optional[bool] = _defaults_clinlp_ner["pseudo"],
    ):
        self.nlp = nlp
        self.attr = attr

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
    def _use_phrase_matcher(self):
        return all(
            self.term_args[field] == _defaults_clinlp_ner[field]
            for field in _non_phrase_matcher_fields
            if field in self.term_args
        )

    def load_concepts(self, concepts: str | dict):
        for concept, concept_terms in concepts.items():
            for concept_term in concept_terms:
                identifier = str(len(self._terms))

                self._terms[identifier] = concept_term
                self._concepts[identifier] = concept

                if isinstance(concept_term, str):
                    if self._use_phrase_matcher:
                        self._phrase_matcher.add(key=identifier, docs=[self.nlp(concept_term)])
                    else:
                        concept_term = Term(concept_term, **self.term_args).to_spacy_pattern(self.nlp)
                        self._matcher.add(key=identifier, patterns=[concept_term])

                elif isinstance(concept_term, list):
                    self._matcher.add(key=identifier, patterns=[concept_term])

                elif isinstance(concept_term, Term):
                    self._matcher.add(key=identifier, patterns=[concept_term.to_spacy_pattern(self.nlp)])

    def _get_matches(self, doc: Doc):
        if len(self._terms) == 0:
            return RuntimeError("No _concepts added.")

        matches = []

        if len(self._matcher) > 0:
            matches += list(self._matcher(doc))

        if len(self._phrase_matcher) > 0:
            matches += list(self._phrase_matcher(doc))

        return matches

    def __call__(self, doc: Doc):
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

        ents = []

        for match_id, start, end in pos_matches:
            if all(
                self._concepts[match_id] != self._concepts[neg_match_id.data]
                for neg_match_id in neg_matches.overlap(start, end)
            ):
                ents.append(Span(doc=doc, start=start, end=end, label=self._concepts[match_id]))

        doc.set_ents(entities=ents)

        return doc