import itertools
from dataclasses import dataclass
from typing import Optional

import intervaltree as ivt
from spacy.language import Doc, Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span

from clinlp.util import clinlp_autocomponent

MAX_LEN = float("inf")


@dataclass
class Term:
    phrase: str
    attr: str = "TEXT"
    proximity: Optional[int] = 0
    fuzzy: Optional[int] = 0
    fuzzy_min_len: Optional[int] = MAX_LEN
    pseudo: Optional[bool] = False

    def to_spacy_pattern(self, nlp: Language):
        spacy_pattern = []
        tokens = [token.text for token in nlp(self.phrase)]

        for token in tokens:
            if (self.fuzzy > 0) and (len(token) < self.fuzzy_min_len):
                token_pattern = {f"FUZZY{self.fuzzy}": token}
            else:
                token_pattern = token

            spacy_pattern.append({self.attr: token_pattern})

            for _ in range(self.proximity):
                spacy_pattern.append({"OP": "?"})

        return spacy_pattern


@clinlp_autocomponent
class ClinlpMatcher:
    concepts: dict
    attr: str = "TEXT"
    fuzzy: Optional[int] = 0
    fuzzy_min_len: Optional[int] = float("inf")

    def __init__(
        self,
        nlp: Language,
        concepts: dict,
        attr: str = "TEXT",
        fuzzy: Optional[int] = 0,
        fuzzy_min_len: Optional[int] = MAX_LEN,
    ):
        self.nlp = nlp
        self.concepts = concepts
        self.attr = attr
        self.fuzzy = fuzzy
        self.fuzzy_min_len = fuzzy_min_len

        self._matcher = Matcher(self.nlp.vocab)
        self._phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=attr)

        self.rules = {}
        self.rule_labels = {}

        self._init_rules(concepts)

    def _init_rules(self, concepts: dict):
        for concept, terms in concepts.items():
            for term in terms:
                identifier = str(len(self.rules))

                self.rules[identifier] = term
                self.rule_labels[identifier] = concept

                if isinstance(term, str):
                    self._phrase_matcher.add(key=identifier, docs=[self.nlp(term)])

                elif isinstance(term, list):
                    self._matcher.add(key=identifier, patterns=[term])

                elif isinstance(term, Term):
                    self._matcher.add(key=identifier, patterns=[term.to_spacy_pattern(self.nlp)])

    def __call__(self, doc: Doc):
        matches = itertools.chain(self._matcher(doc), self._phrase_matcher(doc))

        pos_matches = []
        neg_matches = ivt.IntervalTree()

        for match_id, start, end in matches:
            rule_id = self.nlp.vocab.strings[match_id]
            term = self.rules[rule_id]

            if isinstance(term, Term) and term.pseudo:
                neg_matches[start:end] = rule_id
            else:
                pos_matches.append((rule_id, start, end))

        ents = []

        for match_id, start, end in pos_matches:
            if any(
                self.rule_labels[match_id] == self.rule_labels[neg_match_id.data]
                for neg_match_id in neg_matches.overlap(start, end)
            ):
                continue

            ents.append(Span(doc=doc, start=start, end=end, label=self.rule_labels[match_id]))

        doc.set_ents(entities=ents)

        return doc
