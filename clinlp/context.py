import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Union

import intervaltree as ivt
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span

Span.set_extension(name="qualifiers", default=None)


class Qualifier(Enum):
    ...


class QualifierRuleDirection(Enum):
    PRECEDING = 1
    FOLLOWING = 2
    PSEUDO = 3
    TERMINATION = 4


@dataclass
class QualifierRule:
    pattern: Union[str, list[str]]
    level: Qualifier
    direction: Qualifier


class MatchedQualifierPattern:
    def __init__(self, rule: QualifierRule, start: int, end: int, offset: int = 0):
        self.rule = rule
        self.start = start + offset
        self.end = end + offset
        self.scope = None

    def set_initial_scope(self, sentence: Span):
        if self.rule.direction == QualifierRuleDirection.PRECEDING:
            self.scope = (self.start, sentence.end)

        elif self.rule.direction == QualifierRuleDirection.FOLLOWING:
            self.scope = (sentence.start, self.end)

        else:
            raise ValueError(f"Don't know how to set initial scope of match with rule direction {self.rule.direction}")

    def __repr__(self):
        return {
            "start": self.start,
            "end": self.end,
            "scope": self.scope,
            "rule.pattern": self.rule.pattern,
            "rule.level": str(self.rule.level),
            "rule.direction": str(self.rule.direction),
        }.__repr__()


@Language.factory(name="clinlp_qualifier", requires=["doc.sents"])
class QualifierMatcher:
    def __init__(self, nlp: Language, name: str, phrase_matcher_attr: str = "TEXT"):
        self._nlp = nlp

        self.name = name
        self.rules = {}

        self._matcher = Matcher(self._nlp.vocab, validate=True)
        self._phrase_matcher = PhraseMatcher(self._nlp.vocab, attr=phrase_matcher_attr)

    def add_rule(self, rule: QualifierRule):
        rule_key = f"rule_{len(self.rules)}"
        self.rules[rule_key] = rule

        if isinstance(rule.pattern, str):
            self._phrase_matcher.add(key=rule_key, docs=[self._nlp(rule.pattern)])

        elif isinstance(rule.pattern, list):
            self._matcher.add(key=rule_key, patterns=[rule.pattern])

        else:
            raise ValueError(f"Don't know how to process QualifierRule with pattern of type {type(rule.pattern)}")

    @staticmethod
    def _get_sentences_having_entity(doc: Doc) -> Generator[Span]:
        return (sent for sent in doc.sents if len(sent.ents) > 0)

    def _get_rule_from_match_id(self, match_id: int) -> QualifierRule:
        return self.rules[self._nlp.vocab.strings[match_id]]

    @staticmethod
    def _group_matched_patterns(matched_patterns: list[MatchedQualifierPattern]) -> defaultdict:
        groups = defaultdict(lambda: defaultdict(list))

        for matched_rule in matched_patterns:
            groups[matched_rule.rule.level][matched_rule.rule.direction.name].append(matched_rule)

        return groups

    @staticmethod
    def _limit_scopes_from_terminations(
        scopes: ivt.IntervalTree, terminations: list[MatchedQualifierPattern]
    ) -> ivt.IntervalTree:
        for terminate_match in terminations:
            for interval in scopes.overlap(terminate_match.start, terminate_match.end):
                scopes.remove(interval)
                match = interval.data

                if match.rule.direction == QualifierRuleDirection.PRECEDING and terminate_match.start > match.end:
                    match.scope = (match.scope[0], terminate_match.start)

                if match.rule.direction == QualifierRuleDirection.FOLLOWING and terminate_match.end < match.start:
                    match.scope = (terminate_match.end, match.scope[1])

                scopes[match.scope[0] : match.scope[1]] = match

        return scopes

    def _compute_match_scopes(
        self, matched_patterns: list[MatchedQualifierPattern], sentence: Span
    ) -> ivt.IntervalTree:
        match_scopes = ivt.IntervalTree()

        for _, level_matches in self._group_matched_patterns(matched_patterns).items():
            preceding_ = level_matches[QualifierRuleDirection.PRECEDING.name]
            following_ = level_matches[QualifierRuleDirection.FOLLOWING.name]
            pseudo_ = level_matches[QualifierRuleDirection.PSEUDO.name]
            termination_ = level_matches[QualifierRuleDirection.TERMINATION.name]

            level_matches = ivt.IntervalTree()

            # Following, preceding
            for match in itertools.chain(preceding_, following_):
                match.set_initial_scope(sentence)
                level_matches[match.start : match.end] = match

            # Pseudo
            for match in pseudo_:
                level_matches.remove_overlap(match.start, match.end)

            # Termination
            level_scopes = ivt.IntervalTree(
                ivt.Interval(i.data.scope[0], i.data.scope[1], i.data) for i in level_matches
            )

            match_scopes |= self._limit_scopes_from_terminations(level_scopes, termination_)

        return match_scopes

    def __call__(self, doc: Doc):
        if len(doc.ents) == 0:
            return doc

        for sentence in self._get_sentences_having_entity(doc):
            with warnings.catch_warnings():
                # a UserWarning will trigger when one of the matchers is empty
                warnings.simplefilter("ignore", UserWarning)

                matches = itertools.chain(self._matcher(sentence), self._phrase_matcher(sentence))

            matched_patterns = []

            for match_id, start, end in matches:
                rule = self._get_rule_from_match_id(match_id)
                offset = sentence.start if isinstance(rule.pattern, list) else 0

                matched_patterns.append(
                    MatchedQualifierPattern(
                        rule=self._get_rule_from_match_id(match_id), start=start, end=end, offset=offset
                    )
                )

            match_scopes = self._compute_match_scopes(matched_patterns, sentence)

            for ent in sentence.ents:
                ent._.qualifiers = set()

                for match_interval in match_scopes.overlap(ent.start, ent.end):
                    ent._.qualifiers.add(str(match_interval.data.rule.level))

        return doc
