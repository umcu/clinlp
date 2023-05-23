import itertools
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional, Union

import intervaltree as ivt
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span

QUALIFIERS_ATTR = "qualifiers"


class Qualifier(Enum):
    ...


class QualifierRuleDirection(Enum):
    PRECEDING = 1
    FOLLOWING = 2
    PSEUDO = 3
    TERMINATION = 4


@dataclass
class QualifierRule:
    pattern: Union[str, list[dict[str, str]]]
    level: Qualifier
    direction: QualifierRuleDirection


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

    def __repr__(self):
        return {
            "start": self.start,
            "end": self.end,
            "scope": self.scope,
            "rule.pattern": self.rule.pattern,
            "rule.level": str(self.rule.level),
            "rule.direction": str(self.rule.direction),
        }.__repr__()


def _parse_level(level: str, qualifiers: dict[str, Qualifier]) -> Qualifier:
    if not re.match(r"\w+\.\w+", level):
        raise ValueError(f"Cannot parse level {level}, please adhere to format QualifierClass.level")

    level_class, level = level.split(".")

    return qualifiers[level_class][level]


def _parse_direction(direction: str) -> QualifierRuleDirection:
    return QualifierRuleDirection[direction.upper()]


def load_rules(input_json: Optional[str] = None, data: Optional[dict] = None) -> list[QualifierRule]:
    if input_json and data:
        raise ValueError(
            "Please choose either input_json to load data from json, or provide data as dict, but not both."
        )

    if input_json:
        with open(input_json, "rb") as file:
            data = json.load(file)

    qualifiers = {
        qualifier["qualifier"]: Qualifier(qualifier["qualifier"], qualifier["levels"])
        for qualifier in data["qualifiers"]
    }

    qualifier_rules = []

    for rule in data["rules"]:
        level = _parse_level(rule["level"], qualifiers)
        direction = _parse_direction(rule["direction"])

        qualifier_rules += [QualifierRule(pattern, level, direction) for pattern in rule["patterns"]]

    return qualifier_rules


@Language.factory(
    name="clinlp_qualifier",
    default_config={"phrase_matcher_attr": "TEXT", "qualifiers_attr": QUALIFIERS_ATTR, "rules": None},
    requires=["doc.sents", "doc.ents"],
)
class QualifierMatcher:
    def __init__(
        self,
        nlp: Language,
        name: str,
        phrase_matcher_attr: str = "TEXT",
        qualifiers_attr: str = QUALIFIERS_ATTR,
        rules: Optional[list[QualifierRule]] = None,
    ):
        self.qualifiers_attr = qualifiers_attr

        # TODO: Check if this is the right way to do this?
        if Span.has_extension(qualifiers_attr):
            warnings.warn(
                RuntimeWarning(
                    f"The Span extension {qualifiers_attr} seems already present, please use something"
                    f"else by specifying the qualifiers_attr keyword if this is not intended"
                )
            )

        Span.set_extension(name=qualifiers_attr, default=None, force=True)

        self._nlp = nlp
        self.name = name

        self._matcher = Matcher(self._nlp.vocab, validate=True)
        self._phrase_matcher = PhraseMatcher(self._nlp.vocab, attr=phrase_matcher_attr)

        self.rules = {}

        if rules:
            self.add_rules(rules)

    def add_rule(self, rule: QualifierRule):
        rule_key = f"rule_{len(self.rules)}"
        self.rules[rule_key] = rule

        if isinstance(rule.pattern, str):
            self._phrase_matcher.add(key=rule_key, docs=[self._nlp(rule.pattern)])

        elif isinstance(rule.pattern, list):
            self._matcher.add(key=rule_key, patterns=[rule.pattern])

        else:
            raise ValueError(f"Don't know how to process QualifierRule with pattern of type {type(rule.pattern)}")

    def add_rules(self, rules: list[QualifierRule]):
        for rule in rules:
            self.add_rule(rule)

    def __len__(self):
        return len(self.rules)

    @staticmethod
    def _get_sentences_having_entity(doc: Doc) -> Iterator[Span]:
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

    def _compute_match_scopes(self, matched_patterns: list[MatchedQualifierPattern]) -> ivt.IntervalTree:
        match_scopes = ivt.IntervalTree()

        for _, level_matches in self._group_matched_patterns(matched_patterns).items():
            preceding_ = level_matches[QualifierRuleDirection.PRECEDING.name]
            following_ = level_matches[QualifierRuleDirection.FOLLOWING.name]
            pseudo_ = level_matches[QualifierRuleDirection.PSEUDO.name]
            termination_ = level_matches[QualifierRuleDirection.TERMINATION.name]

            level_matches = ivt.IntervalTree()

            # Following, preceding
            for match in itertools.chain(preceding_, following_):
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

        if len(self.rules) == 0:
            raise RuntimeError("Cannot match qualifiers without any QualifierRule.")

        for sentence in self._get_sentences_having_entity(doc):
            with warnings.catch_warnings():
                # a UserWarning will trigger when one of the matchers is empty
                warnings.simplefilter("ignore", UserWarning)

                matches = itertools.chain(self._matcher(sentence), self._phrase_matcher(sentence))

            matched_patterns = []

            for match_id, start, end in matches:
                rule = self._get_rule_from_match_id(match_id)

                # spacy Matcher handles offset differently than PhraseMatcher, when applying the matcher to a sentence
                offset = sentence.start if isinstance(rule.pattern, list) else 0

                pattern = MatchedQualifierPattern(
                    rule=self._get_rule_from_match_id(match_id), start=start, end=end, offset=offset
                )

                pattern.set_initial_scope(sentence)
                matched_patterns.append(pattern)

            match_scopes = self._compute_match_scopes(matched_patterns)

            for ent in sentence.ents:
                qualifiers = set()

                for match_interval in match_scopes.overlap(ent.start, ent.end):
                    qualifiers.add(str(match_interval.data.rule.level))

                ent._.set(self.qualifiers_attr, qualifiers)

        return doc
