""" Implements qualifier detection for entities. The Qualifier class is reusable. The other classes implement
the context algorithm (https://doi.org/10.1016%2Fj.jbi.2009.05.002) """

import importlib
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
PHRASE_MATCHER_ATTR = "TEXT"

DEFAULT_CONTEXT_RULES = "psynlp_context_rules.json"


class Qualifier(Enum):
    """
    A qualifier modifies an entity (e.g. negation, temporality, plausibility, etc.).
    """

    ...


class ContextRuleDirection(Enum):
    """
    Direction of a Context rule, as in the original algorithm.
    """

    PRECEDING = 1
    FOLLOWING = 2
    PSEUDO = 3
    TERMINATION = 4


@dataclass
class ContextRule:
    """
    A Context rule, as in the original algorithm.

    Args:
        pattern: The pattern to look for in text, either a string, or a spacy pattern (list).
        qualifier: The qualifier to modify.
        direction: The context rule direction.

    """

    pattern: Union[str, list[dict[str, str]]]
    qualifier: Qualifier
    direction: ContextRuleDirection


class _MatchedContextPattern:
    """
    A matched context pattern, that should be processed further.
    """

    def __init__(self, rule: ContextRule, start: int, end: int, offset: int = 0):
        self.rule = rule
        self.start = start + offset
        self.end = end + offset
        self.scope = None

    def set_initial_scope(self, sentence: Span):
        if self.rule.direction == ContextRuleDirection.PRECEDING:
            self.scope = (self.start, sentence.end)

        elif self.rule.direction == ContextRuleDirection.FOLLOWING:
            self.scope = (sentence.start, self.end)


def _parse_qualifier(qualifier: str, qualifier_classes: dict[str, Qualifier]) -> Qualifier:
    """
    Parse a Qualifier from string.

    Args:
        qualifier: The qualifier (e.g. Negation.NEGATED).
        qualifier_classes: A mapping of string to qualifier class.

    Returns: A qualifier, as specified.
    """

    match_regexp = r"\w+\.\w+"

    if not re.match(match_regexp, qualifier):
        raise ValueError(
            f"Cannot parse qualifier {qualifier}, please adhere to format "
            f"{match_regexp} (e.g. NegationQualifier.NEGATED)"
        )

    qualifier_class, qualifier = qualifier.split(".")

    return qualifier_classes[qualifier_class][qualifier]


def _parse_direction(direction: str) -> ContextRuleDirection:
    """
    Parse a Context direction.

    Args:
        direction: The direction.

    Returns: THe ContextRuleDirection.
    """
    return ContextRuleDirection[direction.upper()]


def parse_rules(input_json: Optional[str] = None, data: Optional[dict] = None) -> list[ContextRule]:
    if input_json and data:
        raise ValueError(
            "Please choose either input_json to load data from json, " "or provide data as dict, but not both."
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
        qualifier = _parse_qualifier(rule["qualifier"], qualifiers)
        direction = _parse_direction(rule["direction"])

        qualifier_rules += [ContextRule(pattern, qualifier, direction) for pattern in rule["patterns"]]

    return qualifier_rules


@Language.factory(
    name="clinlp_context_matcher",
    default_config={"phrase_matcher_attr": PHRASE_MATCHER_ATTR, "rules": None},
    requires=["doc.sents", "doc.ents"],
)
class ContextMatcher:
    """
    Implements a very simple version of the context algorithm.

    Args:
        nlp: The Spacy language object to use
        name: The name of the component
        phrase_matcher_attr: The token attribute to match phrases on (e.g. TEXT, ORTH, NORM).
        default_rules: A filename in clinlp.resources with a set of rules to load by default
        rules: A list of ContextRule
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        phrase_matcher_attr: str = PHRASE_MATCHER_ATTR,
        default_rules: Optional[str] = DEFAULT_CONTEXT_RULES,
        rules: Optional[list[ContextRule]] = None,
    ):
        self._nlp = nlp
        self.name = name

        self._matcher = Matcher(self._nlp.vocab, validate=True)
        self._phrase_matcher = PhraseMatcher(self._nlp.vocab, attr=phrase_matcher_attr)

        self.rules = {}

        if default_rules is not None:
            self._load_default_rules(default_rules)

        if rules:
            self.add_rules(rules)

    def _load_default_rules(self, default_rules: str):
        with importlib.resources.path("clinlp.resources", default_rules) as path:
            self.add_rules(parse_rules(path))

    def add_rule(self, rule: ContextRule):
        """
        Add a rule.
        """
        rule_key = f"rule_{len(self.rules)}"
        self.rules[rule_key] = rule

        if isinstance(rule.pattern, str):
            self._phrase_matcher.add(key=rule_key, docs=[self._nlp(rule.pattern)])

        elif isinstance(rule.pattern, list):
            self._matcher.add(key=rule_key, patterns=[rule.pattern])

        else:
            raise ValueError(f"Don't know how to process ContextRule with pattern of type {type(rule.pattern)}")

    def add_rules(self, rules: list[ContextRule]):
        """
        Add multiple rules.
        """
        for rule in rules:
            self.add_rule(rule)

    def __len__(self):
        return len(self.rules)

    @staticmethod
    def _get_sentences_having_entity(doc: Doc) -> Iterator[Span]:
        """
        Return sentences in a doc that have at least one entity.
        """
        return (sent for sent in doc.sents if len(sent.ents) > 0)

    def _get_rule_from_match_id(self, match_id: int) -> ContextRule:
        """
        Get the rule that was matched, from the match_id (first element of match tuple returned by matcher).
        """
        return self.rules[self._nlp.vocab.strings[match_id]]

    @staticmethod
    def _group_matched_patterns(matched_patterns: list[_MatchedContextPattern]) -> defaultdict:
        """
        Group matched patterns by qualifier and direction.
        """
        groups = defaultdict(lambda: defaultdict(list))

        for matched_rule in matched_patterns:
            groups[matched_rule.rule.qualifier][matched_rule.rule.direction.name].append(matched_rule)

        return groups

    @staticmethod
    def _limit_scopes_from_terminations(
        scopes: ivt.IntervalTree, terminations: list[_MatchedContextPattern]
    ) -> ivt.IntervalTree:
        """
        Determine the scope of terminating matched context pattern, return them as IntervalTree.
        """

        for terminate_match in terminations:
            for interval in scopes.overlap(terminate_match.start, terminate_match.end):
                scopes.remove(interval)
                match = interval.data

                if match.rule.direction == ContextRuleDirection.PRECEDING and terminate_match.start > match.end:
                    match.scope = (match.scope[0], terminate_match.start)

                if match.rule.direction == ContextRuleDirection.FOLLOWING and terminate_match.end < match.start:
                    match.scope = (terminate_match.end, match.scope[1])

                scopes[match.scope[0] : match.scope[1]] = match

        return scopes

    def _compute_match_scopes(self, matched_patterns: list[_MatchedContextPattern]) -> ivt.IntervalTree:
        """
        Compute the scope for each matched pattern, return them as an IntervalTree.
        """
        match_scopes = ivt.IntervalTree()

        for _, qualifier_matches in self._group_matched_patterns(matched_patterns).items():
            preceding = qualifier_matches[ContextRuleDirection.PRECEDING.name]
            following = qualifier_matches[ContextRuleDirection.FOLLOWING.name]
            pseudo = qualifier_matches[ContextRuleDirection.PSEUDO.name]
            termination_ = qualifier_matches[ContextRuleDirection.TERMINATION.name]

            qualifier_matches = ivt.IntervalTree()

            # Following, preceding
            for match in preceding + following:
                qualifier_matches[match.start : match.end] = match

            # Pseudo
            for match in pseudo:
                qualifier_matches.remove_overlap(match.start, match.end)

            # Termination
            qualifier_scopes = ivt.IntervalTree(
                ivt.Interval(i.data.scope[0], i.data.scope[1], i.data) for i in qualifier_matches
            )

            match_scopes |= self._limit_scopes_from_terminations(qualifier_scopes, termination_)

        return match_scopes

    def __call__(self, doc: Doc):
        """
        Apply the context matcher to a doc.
        """

        if len(doc.ents) == 0:
            return doc

        if len(self.rules) == 0:
            raise RuntimeError("Cannot match qualifiers without any ContextRule.")

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

                pattern = _MatchedContextPattern(
                    rule=self._get_rule_from_match_id(match_id), start=start, end=end, offset=offset
                )

                pattern.set_initial_scope(sentence)
                matched_patterns.append(pattern)

            match_scopes = self._compute_match_scopes(matched_patterns)

            for ent in sentence.ents:
                qualifiers = set()

                for match_interval in match_scopes.overlap(ent.start, ent.end):
                    qualifiers.add(str(match_interval.data.rule.qualifier))

                ent._.set(QUALIFIERS_ATTR, qualifiers)

        return doc
