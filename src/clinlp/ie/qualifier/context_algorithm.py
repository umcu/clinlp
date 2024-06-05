"""The Context Algorithm as a ``spaCy`` pipeline."""

import importlib.resources
import itertools
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import intervaltree as ivt
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span

from clinlp.ie.qualifier.qualifier import (
    ATTR_QUALIFIERS,
    Qualifier,
    QualifierClass,
    QualifierDetector,
)
from clinlp.util import clinlp_component, interval_dist

_RESOURCES_DIR = importlib.resources.files("clinlp.resources")
_DEFAULT_CONTEXT_RULES_FILE = "context_rules.json"


class ContextRuleDirection(Enum):
    """
    Direction of a rule, as in the original Context Algorithm.

    ``PRECEDING`` means the trigger precedes the entity, while ``FOLLOWING`` means
    it follows the entity. ``BIDIRECTIONAL`` means the trigger can be on either side.
    """

    PRECEDING = 1
    FOLLOWING = 2
    BIDIRECTIONAL = 3
    PSEUDO = 4
    TERMINATION = 5


@dataclass
class ContextRule:
    """
    A Context rule, as in the original Context Algorithm.

    Parameters
    ----------
    pattern
        The pattern to look for in text. Either a ``string``, or a ``spaCy`` pattern 
        (``list``).
    qualifier
        The qualifier to apply.
    direction
        The Context rule direction.
    max_scope
        The maximum scope (number of tokens) of the trigger, or ``None`` for using
        sentence boundaries.
    """

    pattern: Union[str, list[dict[str, str]]]
    qualifier: Qualifier
    direction: ContextRuleDirection
    max_scope: Optional[int] = None


class _MatchedContextPattern:
    """
    A matched Context pattern, that should be processed further.

    Parameters
    ----------
    rule
        The rule that was matched.
    start
        The start index of the match.
    end
        The end index of the match.
    offset
        The offset to apply to the start and end indices.
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
        Set the scope this pattern ranges over, based on the sentence.

        This is either the window determined in the ``max_scope`` of the rule, or the
        sentence boundaries if no ``max_scope`` is set.

        Parameters
        ----------
        sentence
            The sentence the pattern was matched in.
        """
        max_scope = self.rule.max_scope or len(sentence)

        if max_scope < 1:
            msg = f"max_scope must be at least 1, but got {max_scope}"
            raise ValueError(msg)

        scoped_start = max(self.start - max_scope, sentence.start)
        scoped_end = min(self.end + max_scope, sentence.end)

        if self.rule.direction == ContextRuleDirection.PRECEDING:
            self.scope = (self.start, scoped_end)

        elif self.rule.direction == ContextRuleDirection.FOLLOWING:
            self.scope = (scoped_start, self.end)

        elif self.rule.direction == ContextRuleDirection.BIDIRECTIONAL:
            self.scope = (scoped_start, scoped_end)


_defaults_context_algorithm = {
    "phrase_matcher_attr": "TEXT",
    "load_rules": True,
    "rules": str(_RESOURCES_DIR.joinpath(_DEFAULT_CONTEXT_RULES_FILE)),
}


@clinlp_component(
    name="clinlp_context_algorithm",
    requires=["doc.sents", "doc.spans"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_context_algorithm,
)
class ContextAlgorithm(QualifierDetector):
    """
    Implements the Context algorithm as a ``spaCy`` pipeline component.

    For more information, see the original paper:
    https://doi.org/10.1016%2Fj.jbi.2009.05.002

    Parameters
    ----------
    nlp
        The ``spaCy`` language object to use.
    phrase_matcher_attr
        The token attribute to match phrases on (e.g. ``TEXT``, ``ORTH``, ``NORM``).
    load_rules
        Whether to parse any rules. Set this to ``False`` to use
        ``ContextAlgorithm.add_rules`` to add ``ContextRules`` manually.
    rules
        A dictionary of rules, or a path to a ``json`` containing the rules. See the
        ``clinlp.resources`` dir for an example.
    """

    def __init__(
        self,
        nlp: Language,
        phrase_matcher_attr: str = _defaults_context_algorithm["phrase_matcher_attr"],
        load_rules: bool = _defaults_context_algorithm["load_rules"],  # noqa FBT001
        rules: Optional[Union[str | dict]] = _defaults_context_algorithm["rules"],
        **kwargs,
    ) -> None:
        self._nlp = nlp

        self._matcher = Matcher(self._nlp.vocab)
        self._phrase_matcher = PhraseMatcher(self._nlp.vocab, attr=phrase_matcher_attr)

        self.rules = {}
        self._qualifier_classes = {}

        if load_rules:
            if rules is None:
                msg = (
                    "Did not provide rules. Set `load_rules` to False if you "
                    "want to add `ContextRule` manually."
                )
                raise ValueError(msg)

            parsed_rules = self._parse_rules(rules)
            self.add_rules(parsed_rules)

        super().__init__(**kwargs)

    @property
    def qualifier_classes(self) -> dict[str, QualifierClass]:  # noqa D102
        return self._qualifier_classes

    @staticmethod
    def _parse_qualifier(
        qualifier: str, qualifier_classes: dict[str, QualifierClass]
    ) -> Qualifier:
        """
        Parse a qualifier from a string.

        Parameters
        ----------
        qualifier
            The string to parse.
        qualifier_classes
            The available qualifier classes.

        Returns
        -------
            The qualifier, parsed from the string.

        Raises
        ------
        ValueError
            If the qualifier string cannot be parsed.
        """
        match_regexp = r"\w+\.\w+"

        if not re.match(match_regexp, qualifier):
            msg = (
                f"Cannot parse qualifier {qualifier}, please adhere to format "
                f"{match_regexp} (e.g. NegationQualifier.NEGATED)"
            )
            raise ValueError(msg)

        qualifier_class, qualifier = qualifier.split(".")

        return qualifier_classes[qualifier_class].create(value=qualifier)

    @staticmethod
    def _parse_direction(direction: str) -> ContextRuleDirection:
        """
        Parse a direction from a string.

        Parameters
        ----------
        direction
            The string to parse.

        Returns
        -------
            The direction, parsed from the string.
        """
        return ContextRuleDirection[direction.upper()]

    def _parse_rules(self, rules: Union[str | dict]) -> list[ContextRule]:
        if isinstance(rules, str):
            with Path(rules).open(mode="rb") as file:
                rules = json.load(file)

        for qualifier in rules["qualifiers"]:
            self._qualifier_classes[qualifier["name"]] = QualifierClass(**qualifier)

        qualifier_rules = []

        for rule in rules["rules"]:
            qualifier = self._parse_qualifier(rule["qualifier"], self.qualifier_classes)
            direction = self._parse_direction(rule["direction"])
            max_scope = rule.get("max_scope", None)

            qualifier_rules += [
                ContextRule(pattern, qualifier, direction, max_scope)
                for pattern in rule["patterns"]
            ]

        return qualifier_rules

    def add_rule(self, rule: ContextRule) -> None:
        """
        Add a rule to the Context Algorithm.

        Parameters
        ----------
        rule
            The rule to add.

        Raises
        ------
        TypeError
            If the rule pattern is not a ``string`` or a ``list``.
        """
        rule_key = f"rule_{len(self.rules)}"
        self.rules[rule_key] = rule

        if isinstance(rule.pattern, str):
            self._phrase_matcher.add(key=rule_key, docs=[self._nlp(rule.pattern)])

        elif isinstance(rule.pattern, list):
            self._matcher.add(key=rule_key, patterns=[rule.pattern])

        else:
            msg = (
                f"Don't know how to process ContextRule with pattern of "
                f"type {type(rule.pattern)}"
            )
            raise TypeError(msg)

    def add_rules(self, rules: list[ContextRule]) -> None:
        """
        Add multiple rules to the Context Algorithm.

        Parameters
        ----------
        rules
            The rules to add.
        """
        for rule in rules:
            self.add_rule(rule)

    def _get_sentences_with_entities(self, doc: Doc) -> dict[Span, list[Span]]:
        """
        Group entities by sentence.

        Parameters
        ----------
        doc
            The ``spaCy`` doc to process.

        Returns
        -------
            A dictionary mapping sentences to entities.
        """
        sents = defaultdict(list)

        for ent in doc.spans[self.spans_key]:
            sents[ent.sent].append(ent)

        return sents

    def _get_rule_from_match_id(self, match_id: int) -> ContextRule:
        """
        Get the rule from a match ID.

        This is a bit specific to ``spaCy`` matching internals.

        Parameters
        ----------
        match_id
            The match ID to get the rule for.

        Returns
        -------
            The rule that was matched.
        """
        return self.rules[self._nlp.vocab.strings[match_id]]

    @staticmethod
    def _group_matched_patterns(
        matched_patterns: list[_MatchedContextPattern],
    ) -> defaultdict:
        """
        Group matched patterns by qualifier and direction.

        Parameters
        ----------
        matched_patterns
            The matched patterns to group.

        Returns
        -------
            A dictionary mapping qualifiers to directions to matched patterns.
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
        Limit the scopes of matched patterns based on terminations.

        Parameters
        ----------
        scopes
            The scopes to limit.
        terminations
            The terminations to limit the scopes with.

        Returns
        -------
            The limited scopes.
        """
        for terminate_match in terminations:
            for interval in scopes.overlap(terminate_match.start, terminate_match.end):
                scopes.remove(interval)
                match = interval.data

                if (
                    match.rule.direction != ContextRuleDirection.FOLLOWING
                    and terminate_match.start >= match.end
                ):
                    match.scope = (match.scope[0], terminate_match.start)

                if (
                    match.rule.direction != ContextRuleDirection.PRECEDING
                    and terminate_match.end <= match.start
                ):
                    match.scope = (terminate_match.end, match.scope[1])

                scopes[match.scope[0] : match.scope[1]] = match

        return scopes

    def _compute_match_scopes(
        self, matched_patterns: list[_MatchedContextPattern]
    ) -> ivt.IntervalTree:
        """
        Compute the scopes of matched patterns.

        Parameters
        ----------
        matched_patterns
            The matched patterns to compute scopes for.

        Returns
        -------
            The scopes of the matched patterns.
        """
        match_scopes = ivt.IntervalTree()

        for qualifier_matches in self._group_matched_patterns(
            matched_patterns
        ).values():
            preceding = qualifier_matches[ContextRuleDirection.PRECEDING]
            following = qualifier_matches[ContextRuleDirection.FOLLOWING]
            bidirectional = qualifier_matches[ContextRuleDirection.BIDIRECTIONAL]
            pseudo = qualifier_matches[ContextRuleDirection.PSEUDO]
            termination = qualifier_matches[ContextRuleDirection.TERMINATION]

            qualifier_matches = ivt.IntervalTree()

            # Following, preceding
            for match in preceding + following + bidirectional:
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

    def _resolve_matched_pattern_conflicts(
        self, entity: Span, matched_patterns: list[_MatchedContextPattern]
    ) -> list[_MatchedContextPattern]:
        """
        Resolve conflicts between matched patterns.

        Works finding the pattern with smallest interval distance to the entity,
        followed by the one with the highest priority (in case of ties).

        Parameters
        ----------
        entity
            The entity to resolve conflicts for.
        matched_patterns
            The matched patterns to resolve conflicts for.

        Returns
        -------
            The resolved matched patterns.
        """
        if len(matched_patterns) <= 1:
            return matched_patterns

        grouped_patterns = defaultdict(list)
        result_patterns = []

        for mp in matched_patterns:
            grouped_patterns[mp.rule.qualifier.name].append(mp)

        for mp_group in grouped_patterns.values():
            if len(mp_group) == 1:
                result_patterns += mp_group
            else:
                result_patterns.append(
                    min(
                        mp_group,
                        key=lambda mp: (
                            interval_dist(entity.start, entity.end, mp.start, mp.end),
                            -mp.rule.qualifier.priority,
                        ),
                    )
                )

        return result_patterns

    def _detect_qualifiers(self, doc: Doc) -> None:
        """
        Detect qualifiers in a document.

        Parameters
        ----------
        doc
            The ``spaCy`` doc to process.

        Raises
        ------
        RuntimeError
            If no rules are set.
        """
        if len(self.rules) == 0:
            msg = "Cannot match qualifiers without any ContextRule."
            raise RuntimeError(msg)

        for sentence, ents in self._get_sentences_with_entities(doc).items():
            with warnings.catch_warnings():
                # a UserWarning will trigger when one of the matchers is empty
                warnings.simplefilter("ignore", UserWarning)

                matches = itertools.chain(
                    self._matcher(sentence), self._phrase_matcher(sentence)
                )

            matched_patterns = []

            for match_id, start, end in matches:
                rule = self._get_rule_from_match_id(match_id)

                # spacy Matcher handles offset differently than PhraseMatcher,
                # when applying the matcher to a sentence
                offset = sentence.start if isinstance(rule.pattern, list) else 0

                matched_pattern = _MatchedContextPattern(
                    rule=self._get_rule_from_match_id(match_id),
                    start=start,
                    end=end,
                    offset=offset,
                )

                matched_pattern.initialize_scope(sentence)
                matched_patterns.append(matched_pattern)

            match_scopes = self._compute_match_scopes(matched_patterns)

            for ent in ents:
                matched_patterns = []

                for match_interval in match_scopes.overlap(ent.start, ent.end):
                    if (ent.start + 1 > match_interval.data.end) or (
                        ent.end < match_interval.data.start + 1
                    ):
                        matched_patterns.append(match_interval.data)

                matched_patterns = self._resolve_matched_pattern_conflicts(
                    ent, matched_patterns
                )

                for matched_pattern in matched_patterns:
                    self.add_qualifier_to_ent(ent, matched_pattern.rule.qualifier)

    def __len__(self) -> int:
        """
        Return the number of rules added.

        Returns
        -------
            The number of rules added.
        """
        return len(self.rules)
