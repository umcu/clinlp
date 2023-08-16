from dataclasses import dataclass
from typing import Optional

import intervaltree as ivt
from spacy.language import Doc, Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span

from clinlp.util import clinlp_autocomponent

_defaults_clinlp_ner = {"attr": "TEXT", "proximity": 0, "fuzzy": 0, "fuzzy_min_len": 0, "pseudo": False}


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
    name="clinlp_ner",
    requires=["doc.sents", "doc.ents"],
    assigns=[f"doc.ents"],
    default_config=_defaults_clinlp_ner,
)
@clinlp_autocomponent
class ClinlpNer(Term):
    def __init__(self, nlp: Language, **kwargs):
        self.nlp = nlp
        self.attr = kwargs.get('attr', _defaults_clinlp_ner['attr'])
        self.term_defaults = kwargs

        self._matcher = Matcher(self.nlp.vocab)
        self._phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=self.attr)

        self.terms = {}
        self.term_concept = {}

    @property
    def _use_phrase_matcher(self):
        non_phrase_matcher_settings = ["proximity", "fuzzy", "fuzzy_min_len"]

        for field in non_phrase_matcher_settings:
            if field in self.term_defaults and self.term_defaults[field] != _defaults_clinlp_ner[field]:
                return False

        return True

    def load_concepts(self, concepts: str | dict):
        for concept, concept_terms in concepts.items():
            for concept_term in concept_terms:
                identifier = str(len(self.terms))

                self.terms[identifier] = concept_term
                self.term_concept[identifier] = concept

                if isinstance(concept_term, str):
                    if self._use_phrase_matcher:
                        self._phrase_matcher.add(key=identifier, docs=[self.nlp(concept_term)])
                    else:
                        concept_term = Term(concept_term, **self.term_defaults).to_spacy_pattern(self.nlp)
                        self._matcher.add(key=identifier, patterns=[concept_term])

                elif isinstance(concept_term, list):
                    self._matcher.add(key=identifier, patterns=[concept_term])

                elif isinstance(concept_term, Term):
                    self._matcher.add(key=identifier, patterns=[concept_term.to_spacy_pattern(self.nlp)])

    def _get_matches(self, doc: Doc):

        if len(self.terms) == 0:
            return RuntimeError("No concepts added.")

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
            term = self.terms[rule_id]

            if isinstance(term, Term) and term.pseudo:
                neg_matches[start:end] = rule_id
            else:
                pos_matches.append((rule_id, start, end))

        ents = []

        for match_id, start, end in pos_matches:
            if any(
                self.term_concept[match_id] == self.term_concept[neg_match_id.data]
                for neg_match_id in neg_matches.overlap(start, end)
            ):
                continue

            ents.append(Span(doc=doc, start=start, end=end, label=self.term_concept[match_id]))

        doc.set_ents(entities=ents)

        return doc
