"""Component for rule based entity matching."""

import json
from collections.abc import Iterable
from pathlib import Path

import intervaltree as ivt
import pandas as pd
import pydantic
from spacy.language import Doc, Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import Pipe
from spacy.tokens import Span

from clinlp.ie.term import Term
from clinlp.util import clinlp_component

SPANS_KEY = "ents"


@clinlp_component(
    name="clinlp_rule_based_entity_matcher",
    assigns=["doc.spans"],
)
class RuleBasedEntityMatcher(Pipe):
    """
    ``spaCy`` component for rule-based entity matching.

    This component can be used to match entities based on known concepts, along with
    terms/synonyms to match (per concept). It can do literal string matching, but also
    has some additional configuration options like fuzzy matching and proximity
    matching. Note that configuration (e.g. ``attr``, ``proximity``, ...) set at the
    entity matcher level is overridden by the configuration at the term level.
    """

    _non_phrase_matcher_fields = ("proximity", "fuzzy", "fuzzy_min_len")

    def __init__(
        self,
        nlp: Language,
        *,
        attr: str = "TEXT",
        proximity: int = 0,
        fuzzy: int = 0,
        fuzzy_min_len: int = 0,
        pseudo: bool = False,
        resolve_overlap: bool = False,
        spans_key: str = SPANS_KEY,
    ) -> None:
        """
        Create a rule-based entity matcher.

        Parameters
        ----------
        nlp
            The ``spaCy`` language model.
        attr
            The attribute to match on.
        proximity
            The number of tokens to allow between each token in the phrase.
        fuzzy
            The threshold for fuzzy matching.
        fuzzy_min_len
            The minimum length for fuzzy matching.
        pseudo
            Whether this term is a pseudo-term, which is excluded from matches.
        resolve_overlap
            Whether to resolve overlapping entities.
        spans_key
            The key to store the entities in the document.
        """
        self.nlp = nlp

        self.term_args = {
            "attr": attr,
            "proximity": proximity,
            "fuzzy": fuzzy,
            "fuzzy_min_len": fuzzy_min_len,
            "pseudo": pseudo,
        }

        self.resolve_overlap = resolve_overlap
        self.spans_key = spans_key

        self._phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=attr)
        self._matcher = Matcher(self.nlp.vocab)

        self._terms = {}
        self._concepts = {}

    def add_term(self, concept: str, term: str | dict | list | Term) -> None:
        """
        Add a term for matching, along with a concept identifier.

        Note that concepts do not need to be added separately. It's is also possible to
        call `add_term` multiple with the same concept identifier (terms will be
        appended, not overwritten).

        Parameters
        ----------
        concept
            The concept identifier.
        term
            The term that should be matched. Can be a string (i.e. a phrase), a dict
            (that is passed directly to the ``clinlp.ie.Term`` constructor), a list
            comprising a ``spaCy`` pattern, or a ``clinlp.ie.Term`` object.

        Raises
        ------
        TypeError
            If the term type is not supported.
        """
        allowed_types = (str, dict, list, Term)

        if not any(isinstance(term, allowed_type) for allowed_type in allowed_types):
            msg = (
                f"The term type {type(term)} is not supported. Please provide a "
                "string, dict, list, or Term object."
            )

            raise TypeError(msg)

        matcher_key = str(len(self._terms))

        self._terms[matcher_key] = term
        self._concepts[matcher_key] = concept

        if isinstance(term, str):
            term = Term(phrase=term)

        if isinstance(term, dict):
            term = Term(**term)

        if isinstance(term, list):
            self._matcher.add(key=matcher_key, patterns=[term])

        if isinstance(term, Term):
            term.override_non_set_fields(self.term_args)
            term_defaults = Term.defaults()

            if term.attr == self.term_args["attr"] and all(
                getattr(term, field) == term_defaults[field]
                for field in self._non_phrase_matcher_fields
            ):
                doc = self.nlp(term.phrase)
                self._phrase_matcher.add(key=matcher_key, docs=[doc])

            else:
                pattern = term.to_spacy_pattern(self.nlp)
                self._matcher.add(key=matcher_key, patterns=[pattern])

    def add_terms(
        self, concept: str, terms: Iterable[str | dict | list | Term]
    ) -> None:
        """
        Add multiple terms with the same concept identifier.

        Parameters
        ----------
        concept
            A concept identifier, applicable to all terms.
        terms
            An iterable of terms to add.
        """
        for term in terms:
            self.add_term(concept=concept, term=term)

    def add_terms_from_dict(
        self, terms: dict[str, Iterable[str | dict | list | Term]]
    ) -> None:
        """
        Add terms from a dictionary.

        The dictionary should have the concept identifier as the key, and a list of
        terms as values.

        Parameters
        ----------
        data
            The concepts and terms in dictionary form.
        """
        for concept, concept_terms in terms.items():
            self.add_terms(concept=concept, terms=concept_terms)

    def add_terms_from_json(self, path: str) -> None:
        """
        Add terms from a JSON file.

        The JSON file should have a "terms" key containing the terms and concepts. This
        dictionary should have the concept identifier as the key, and a list of terms
        as values.

        Parameters
        ----------
        path
            The path to the JSON file.

        Raises
        ------
        ValueError
            If a 'terms' key is not found in the JSON file.
        """
        with Path(path).open() as f:
            data = json.load(f)

        if "terms" not in data:
            msg = 'Please provide a JSON file with a "concepts" key.'

            raise ValueError(msg)

        self.add_terms_from_dict(terms=data["terms"])

    def add_terms_from_csv(
        self, path: str, concept_col: str = "concept", **kwargs
    ) -> None:
        """
        Add concepts from a csv file.

        The csv should contain the concept identifier in the "concept_col" column,
        and the term arguments as columns. Must at least include a column for the
        phrase, and optionally other columns for the clinlp.ie.Term arguments.
        Any other columns are ignored.

        Parameters
        ----------
        path
            A path to the csv file.
        concept_col, optional
            The column name for the concept identifier.
        **kwargs
            Any additional keyword arguments to pass to the ``pandas.read_csv`` method.

        Raises
        ------
        RuntimeError
            If a value in the csv file cannot be parsed.
        """
        df = pd.read_csv(path, **kwargs)

        for _, row in df.iterrows():
            try:
                term_args = {k: v for k, v in row.to_dict().items() if not pd.isna(v)}
                term = Term(**term_args)
            except pydantic.ValidationError as e:
                msg = (
                    "There is a value in your input csv which cannot be"
                    "parsed. Please refer to the above error for more details."
                )

                raise RuntimeError(msg) from e

            self.add_term(concept=row[concept_col], term=term)

    def _get_matches(self, doc: Doc) -> list[tuple[int, int, int]]:
        """
        Get the matches from the matcher and phrase matcher.

        Parameters
        ----------
        doc
            The document.

        Returns
        -------
        ``list[tuple[int, int, int]]``
            The matches.

        Raises
        ------
        RuntimeError
            If no concepts have been added.
        """
        if len(self._terms) == 0:
            msg = "No concepts added."
            raise RuntimeError(msg)

        matches = []

        if len(self._phrase_matcher) > 0:
            matches += list(self._phrase_matcher(doc))

        if len(self._matcher) > 0:
            matches += list(self._matcher(doc))

        return matches

    @staticmethod
    def _resolve_ents_overlap(ents: list[Span]) -> list[Span]:
        """
        Resolve overlap between entities.

        Takes the longest entity in case of overlap.

        Parameters
        ----------
        ents
            The entities.

        Returns
        -------
        ``list[Span]``
            The entities, no longer overlapping.
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

    def match_entities(self, doc: Doc) -> list[Span]:
        """
        Match entities in a document.

        Parameters
        ----------
        doc
            The document.

        Returns
        -------
        ``list[Span]``
            The entities.
        """
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

        return ents

    def __call__(self, doc: Doc) -> Doc:
        """
        Match entities in a document text and add to document.

        The entities that are found will be stored in ``doc.spans['ents']``. Make sure
        any subsequent components expect the entities to be stored there.

        Parameters
        ----------
        doc
            The document.

        Returns
        -------
        ``Doc``
            The document with entities.
        """
        ents = self.match_entities(doc)

        doc.spans[self.spans_key] = ents

        return doc
