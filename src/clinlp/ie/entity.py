"""Component for rule based entity matching."""

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


def create_concept_dict(path: str, concept_col: str = "concept") -> dict:
    """
    Create a dictionary of concepts and their terms from a ``csv`` file.

    The resulting dictionary can be passed directly into the ``load_concepts`` method
    of the ``RuleBasedEntityMatcher``.

    Parameters
    ----------
    path
        The path to the ``csv`` file.
    concept_col
        The column containing the concept identifier.

    Returns
    -------
    ``dict``
        A dictionary of concepts and their terms.

    Raises
    ------
    RuntimeError
        If a value in the input ``csv`` cannot be parsed.
    """
    df = pd.read_csv(path)

    try:
        df["term"] = df.apply(
            lambda x: Term(**{k: v for k, v in x.to_dict().items() if not pd.isna(v)}),
            axis=1,
        )
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
    """Deprecated, use ``clinlp_rule_based_entity_matcher`` instead."""

    def __init__(self) -> None:
        msg = (
            "The clinlp_entity_matcher has been renamed "
            "clinlp_rule_based_entity_matcher."
        )

        raise RuntimeError(msg)


@clinlp_component(
    name="clinlp_rule_based_entity_matcher",
    assigns=["doc.spans"],
)
class RuleBasedEntityMatcher(Pipe):
    """
    ``spaCy`` component for rule-based entity matching.

    This component is used to match entities based on a set of concepts, along with
    synonyms. Note that settings (e.g. ``attr``, ``proximity``, ...) set at the entity
    matcher level are overridden by the settings at the term level.
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

        self._matcher = Matcher(self.nlp.vocab)
        self._phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=attr)

        self._terms = {}
        self._concepts = {}

    @property
    def _use_phrase_matcher(self) -> bool:
        """
        Determine whether the ``spaCy`` phrase matcher can be used.

        This is the case if all term arguments are set to their default values, so no
        complex ``spaCy`` patterns are required.

        Returns
        -------
        ``bool``
            Whether the phrase matcher can be used.
        """
        term_defaults = Term.defaults()

        return all(
            self.term_args[field] == term_defaults[field]
            for field in self._non_phrase_matcher_fields
        )

    def load_concepts(self, concepts: dict) -> None:
        """
        Load a dictionary of concepts and their terms.

        Parameters
        ----------
        concepts
            A dictionary of concepts and their terms. Present with concepts as keys,
            and lists of terms as values. Each term can be a ``string``, a ``spaCy``
            pattern, or a ``clinlp.Term``.

        Raises
        ------
        TypeError
            If the term type is not ``str``, ``list`` or ``clinlp.Term``.
        """
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
                        if field in concept_term.fields_set:
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
                        f"please provide str, list or clinlp.ie.Term."
                    )
                    raise TypeError(msg)

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

        if len(self._matcher) > 0:
            matches += list(self._matcher(doc))

        if len(self._phrase_matcher) > 0:
            matches += list(self._phrase_matcher(doc))

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

    def __call__(self, doc: Doc) -> Doc:
        """
        Find entities in a document text.

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

        doc.spans[self.spans_key] = ents

        return doc
