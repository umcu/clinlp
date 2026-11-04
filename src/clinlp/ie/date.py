"""Associate explicit calendar dates with existing entity spans."""

import re
from datetime import date

from spacy.pipeline import Pipe
from spacy.tokens import Doc, Span

from clinlp.ie.entity import SPANS_KEY
from clinlp.util import clinlp_component

_DATE_PATTERN = re.compile(
    r"(?<![\w/.-])(?:"
    r"(?P<iso_year>[0-9]{4})-(?P<iso_month>[0-9]{2})-(?P<iso_day>[0-9]{2})"
    r"|(?P<day>[0-9]{1,2})(?P<separator>[/.-])"
    r"(?P<month>[0-9]{1,2})(?P=separator)(?P<year>[0-9]{4})"
    r"|(?P<ym_year>[0-9]{4})[-/](?P<ym_month>[0-9]{1,2})"
    r"|(?P<my_month>[0-9]{1,2})[-/](?P<my_year>[0-9]{4})"
    r"|(?P<first>[0-9]{1,2})[-/](?P<second>[0-9]{1,2})"
    r")(?!\w|[./-][0-9])"
)

if not Span.has_extension("date"):
    Span.set_extension("date", default=None)


@clinlp_component(
    name="clinlp_date_linker",
    requires=["doc.spans"],
    assigns=["span._.date"],
)
class DateLinker(Pipe):
    """Link entities to full or partial numeric dates on the same line.

    Missing years and uncertain day/month order are retained as ambiguous metadata,
    not resolved by guessing. This structural heuristic does not determine whether
    a date describes an event,
    a report, or another temporal role. It leaves multiple-date lines unresolved and
    never falls back to dates in other lines or to document metadata.
    """

    def __init__(
        self,
        *,
        spans_key: str = SPANS_KEY,
        max_distance: int = 200,
    ) -> None:
        """Create a date linker.

        Parameters
        ----------
        spans_key
            The document span group containing the extracted concepts.
        max_distance
            Maximum character gap between the entity and date, inclusive. Dates
            must also be on the same physical line as the entire entity.

        Raises
        ------
        ValueError
            If ``max_distance`` is negative.
        """
        if max_distance < 0:
            msg = "max_distance must be non-negative."
            raise ValueError(msg)

        self.spans_key = spans_key
        self.max_distance = max_distance

    @staticmethod
    def _valid_short_date(first: int, second: int) -> bool:
        """Accept either day/month order without assuming a year or leap status."""
        for month, day in ((first, second), (second, first)):
            try:
                # A leap year allows February 29 without inferring an event year.
                date(2000, month, day)
            except ValueError:
                continue
            return True
        return False

    @classmethod
    def _normalize(cls, match: re.Match) -> tuple[str | None, str] | None:
        """Normalize known components and explicitly retain unresolved dates."""
        if match.group("first") is not None:
            if cls._valid_short_date(int(match["first"]), int(match["second"])):
                return None, "ambiguous"
            return None

        if match.group("ym_year") is not None or match.group("my_year") is not None:
            year = int(match["ym_year"] or match["my_year"])
            month = int(match["ym_month"] or match["my_month"])
            if 1 <= year <= 9999 and 1 <= month <= 12:
                return f"{year:04d}-{month:02d}", "month"
            return None

        if match.group("iso_year") is not None:
            parts = match.group("iso_year", "iso_month", "iso_day")
        else:
            parts = match.group("year", "month", "day")

        try:
            return date(*(int(part) for part in parts)).isoformat(), "day"
        except ValueError:
            return None

    def _find_date(self, entity: Span) -> dict | None:
        """Find a single non-overlapping date on the entity's physical line."""
        text = entity.doc.text
        if "\n" in entity.text or "\r" in entity.text or not entity.text:
            return None

        line_start = (
            max(
                text.rfind("\n", 0, entity.start_char),
                text.rfind("\r", 0, entity.start_char),
            )
            + 1
        )
        line_end = min(
            position
            for position in (
                text.find("\n", entity.end_char),
                text.find("\r", entity.end_char),
                len(text),
            )
            if position != -1
        )
        matches = list(_DATE_PATTERN.finditer(text, line_start, line_end))

        # Check the whole line before applying proximity: a second date may have
        # another temporal role, even when it is farther away or invalid.
        if len(matches) != 1:
            return None

        match = matches[0]
        if match.start() < entity.end_char and match.end() > entity.start_char:
            return None

        distance = max(entity.start_char - match.end(), match.start() - entity.end_char)
        if distance > self.max_distance:
            return None

        normalized = self._normalize(match)
        if normalized is None:
            return None
        value, precision = normalized

        return {
            "value": value,
            "text": match.group(),
            "start_char": match.start(),
            "end_char": match.end(),
            "precision": precision,
            "method": "same_line",
        }

    def __call__(self, doc: Doc) -> Doc:
        """Annotate existing spans without changing their text, labels or qualifiers.

        Each span's ``_.date`` is replaced with a serializable dictionary containing
        the date's precision and provenance, or ``None`` if no date can be linked.
        Values are ISO dates for day precision, YYYY-MM for month precision, and
        ``None`` for ambiguous expressions; their original text is retained.
        Character offsets refer to the original ``doc.text`` and are end-exclusive.
        A missing span group is treated as empty.
        """
        for entity in doc.spans.get(self.spans_key, ()):
            entity._.date = self._find_date(entity)

        return doc
