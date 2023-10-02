import importlib.resources
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

from clinlp.qualifier.qualifier import (
    ATTR_QUALIFIERS,
    Qualifier,
    QualifierDetector,
    QualifierFactory,
)
from clinlp.util import clinlp_autocomponent


class ContextRuleDirection(Enum):
    """
    Direction of a Context rule, as in the original Context Algorithm.
    """

    PRECEDING = 1
    FOLLOWING = 2
    PSEUDO = 3
    TERMINATION = 4


@dataclass
class ContextRule:
    """
    A Context rule, as in the original Context Algorithm.

    Args:
        pattern: The pattern to look for in text. Either a string, or a spacy pattern (list).
        qualifier: The qualifier to apply.
        direction: The Context rule direction.
        max_scope: The maximum scope (number of tokens) of the trigger, or None for using sentence boundaries.
    """

    pattern: Union[str, list[dict[str, str]]]
    qualifier: Qualifier
    direction: ContextRuleDirection
    max_scope: Optional[int] = None


class _MatchedContextPattern:
    """
    A matched Context pattern, that should be processed further.
    """

    def __init__(
        self, rule: ContextRule, start: int, end: int, offset: int = 0
    ) -> None:
        self.rule = rule
        self.start = start + offset
        self.end = end + offset
        self.scope = None

    def initialize_scope(self, sentence: Span) -> None:
        """
        Sets the scope this pattern ranges over, based on the sentence. This is either the window determined in
        the `max_scope` of the rule, or the sentence boundaries if no `max_scope` is set.
        """

        max_scope = self.rule.max_scope or len(sentence)

        if max_scope < 1:
            raise ValueError(f"max_scope must be at least 1, but got {max_scope}")

        if self.rule.direction == ContextRuleDirection.PRECEDING:
            self.scope = (self.start, min(self.end + max_scope, sentence.end))

        elif self.rule.direction == ContextRuleDirection.FOLLOWING:
            self.scope = (max(self.start - max_scope, sentence.start), self.end)


_defaults_context_algorithm = {
    "phrase_matcher_attr": "TEXT",
    "load_rules": True,
    "rules": str(importlib.resources.path("clinlp.resources", "context_rules.json")),
}


@Language.factory(
    name="clinlp_context_algorithm",
    requires=["doc.sents", "doc.ents"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_context_algorithm,
)
@clinlp_autocomponent
class ContextAlgorithm(QualifierDetector):
    """
    Implements the Context algorithm (https://doi.org/10.1016%2Fj.jbi.2009.05.002) as a spaCy pipeline component.

    Args:
        nlp: The Spacy language object to use
        phrase_matcher_attr: The token attribute to match phrases on (e.g. TEXT, ORTH, NORM).
        load_rules: Whether to parse any rules. Set this to `False` to use ContextAlgorithm.add_rules to
        rules: A dictionary of rules, or a path to a json containing the rules (see clinlp.resources dir for example).
        add ContextRules manually.
    """

    def __init__(
        self,
        nlp: Language,
        phrase_matcher_attr: str = _defaults_context_algorithm["phrase_matcher_attr"],
        load_rules=_defaults_context_algorithm["load_rules"],
        rules: Optional[Union[str | dict]] = _defaults_context_algorithm["rules"],
    ) -> None:
        self._nlp = nlp

        self._matcher = Matcher(self._nlp.vocab)
        self._phrase_matcher = PhraseMatcher(self._nlp.vocab, attr=phrase_matcher_attr)

        self.rules = {}
        self._qualifier_factories = {}

        if load_rules:
            if rules is None:
                raise ValueError(
                    "Did not provide rules. Set `load_rules` to False if you want to add `ContextRule` manually."
                )

            rules = self._parse_rules(rules)
            self.add_rules(rules)

    @property
    def qualifier_factories(self) -> dict[str, QualifierFactory]:
        return self._qualifier_factories

    def add_rule(self, rule: ContextRule) -> None:
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
            raise ValueError(
                f"Don't know how to process ContextRule with pattern of type {type(rule.pattern)}"
            )

    def add_rules(self, rules: list[ContextRule]) -> None:
        """
        Add multiple rules.
        """
        for rule in rules:
            self.add_rule(rule)

    @staticmethod
    def _parse_qualifier(
        qualifier: str, qualifier_factories: dict[str, QualifierFactory]
    ) -> Qualifier:
        """
        Parse a Qualifier from string.

        Args:
            qualifier: The qualifier (e.g. Negation.NEGATED).

        Returns: A qualifier, as specified.
        """

        match_regexp = r"\w+\.\w+"

        if not re.match(match_regexp, qualifier):
            raise ValueError(
                f"Cannot parse qualifier {qualifier}, please adhere to format "
                f"{match_regexp} (e.g. NegationQualifier.NEGATED)"
            )

        qualifier_class, qualifier = qualifier.split(".")

        return qualifier_factories[qualifier_class].create(value=qualifier)

    @staticmethod
    def _parse_direction(direction: str) -> ContextRuleDirection:
        """
        Parse a Context direction.

        Args:
            direction: The direction (e.g. preceding, following, etc.).

        Returns: THe ContextRuleDirection.
        """
        return ContextRuleDirection[direction.upper()]

    def _parse_rules(self, rules: Union[str | dict]) -> list[ContextRule]:
        if isinstance(rules, str):
            with open(rules, "rb") as file:
                rules = json.load(file)

        for qualifier in rules["qualifiers"]:
            self._qualifier_factories[qualifier["name"]] = QualifierFactory(**qualifier)

        qualifier_rules = []

        for rule in rules["rules"]:
            qualifier = self._parse_qualifier(
                rule["qualifier"], self.qualifier_factories
            )
            direction = self._parse_direction(rule["direction"])
            max_scope = rule.get("max_scope", None)

            qualifier_rules += [
                ContextRule(pattern, qualifier, direction, max_scope)
                for pattern in rule["patterns"]
            ]

        return qualifier_rules

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
    def _group_matched_patterns(
        matched_patterns: list[_MatchedContextPattern],
    ) -> defaultdict:
        """
        Group matched patterns by qualifier and direction.
        """
        groups = defaultdict(lambda: defaultdict(list))

        for matched_rule in matched_patterns:
            groups[matched_rule.rule.qualifier][matched_rule.rule.direction].append(
                matched_rule
            )

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

                if (
                    match.rule.direction == ContextRuleDirection.PRECEDING
                    and terminate_match.start > match.end
                ):
                    match.scope = (match.scope[0], terminate_match.start)

                if (
                    match.rule.direction == ContextRuleDirection.FOLLOWING
                    and terminate_match.end < match.start
                ):
                    match.scope = (terminate_match.end, match.scope[1])

                scopes[match.scope[0] : match.scope[1]] = match

        return scopes

    def _compute_match_scopes(
        self, matched_patterns: list[_MatchedContextPattern]
    ) -> ivt.IntervalTree:
        """
        Compute the scope for each matched pattern, return them as an IntervalTree.
        """
        match_scopes = ivt.IntervalTree()

        for qualifier_matches in self._group_matched_patterns(
            matched_patterns
        ).values():
            preceding = qualifier_matches[ContextRuleDirection.PRECEDING]
            following = qualifier_matches[ContextRuleDirection.FOLLOWING]
            pseudo = qualifier_matches[ContextRuleDirection.PSEUDO]
            termination = qualifier_matches[ContextRuleDirection.TERMINATION]

            qualifier_matches = ivt.IntervalTree()

            # Following, preceding
            for match in preceding + following:
                qualifier_matches[match.start : match.end] = match

            # Pseudo
            for match in pseudo:
                qualifier_matches.remove_overlap(match.start, match.end)

            # Termination
            qualifier_scopes = ivt.IntervalTree(
                ivt.Interval(i.data.scope[0], i.data.scope[1], i.data)
                for i in qualifier_matches
            )

            match_scopes |= self._limit_scopes_from_terminations(
                qualifier_scopes, termination
            )

        return match_scopes

    def _detect_qualifiers(self, doc: Doc):
        """
        Apply the Context Algorithm to a doc.
        """

        if len(self.rules) == 0:
            raise RuntimeError("Cannot match qualifiers without any ContextRule.")

        for sentence in self._get_sentences_having_entity(doc):
            with warnings.catch_warnings():
                # a UserWarning will trigger when one of the matchers is empty
                warnings.simplefilter("ignore", UserWarning)

                matches = itertools.chain(
                    self._matcher(sentence), self._phrase_matcher(sentence)
                )

            matched_patterns = []

            for match_id, start, end in matches:
                rule = self._get_rule_from_match_id(match_id)

                # spacy Matcher handles offset differently than PhraseMatcher, when applying the matcher to a sentence
                offset = sentence.start if isinstance(rule.pattern, list) else 0

                pattern = _MatchedContextPattern(
                    rule=self._get_rule_from_match_id(match_id),
                    start=start,
                    end=end,
                    offset=offset,
                )

                pattern.initialize_scope(sentence)
                matched_patterns.append(pattern)

            match_scopes = self._compute_match_scopes(matched_patterns)

            for ent in sentence.ents:
                for match_interval in match_scopes.overlap(ent.start, ent.end):
                    if (ent.start + 1 > match_interval.data.end) or (
                        ent.end < match_interval.data.start + 1
                    ):
                        self.add_qualifier_to_ent(
                            ent, match_interval.data.rule.qualifier
                        )

    def __len__(self) -> int:
        return len(self.rules)
