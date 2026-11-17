"""Associate explicit calendar dates with existing entity spans."""

import re
from datetime import date

from spacy.pipeline import Pipe
from spacy.tokens import Doc, Span

from clinlp.ie.entity import SPANS_KEY
from clinlp.util import clinlp_component

_HORIZONTAL_SPACE = r"[^\S\r\n\u2028\u2029]"
_SEPARATOR = r"[/.\-\u2010-\u2015\u2212]"
_MONTH_NUMBERS = {
    name: number
    for number, names in enumerate(
        (
            ("januari", "jan"),
            ("februari", "feb", "febr"),
            ("maart", "mrt"),
            ("april", "apr"),
            ("mei",),
            ("juni", "jun"),
            ("juli", "jul"),
            ("augustus", "aug"),
            ("september", "sep", "sept"),
            ("oktober", "okt"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_PATTERN = (
    "(?:" + "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True)) + r")\.?"
)
_TIME_PATTERN = (
    r"[Tt][0-9]{1,2}:[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]+)?)?"
    r"(?:[Zz]|[+-][0-9]{2}:?[0-9]{2})?"
)
_DATE_PATTERN = re.compile(
    rf"(?<![\w/.\-\u2010-\u2015\u2212])(?:"
    rf"(?P<named>(?:[0-9]+{_HORIZONTAL_SPACE}*)?{_MONTH_PATTERN}"
    rf"{_HORIZONTAL_SPACE}*[0-9]+)"
    rf"|(?P<named_short>[0-9]+{_HORIZONTAL_SPACE}*{_MONTH_PATTERN})"
    rf"|(?P<numeric>[0-9]+(?:{_HORIZONTAL_SPACE}*{_SEPARATOR}"
    rf"{_HORIZONTAL_SPACE}*[0-9]+)+)(?:{_TIME_PATTERN})?"
    r"|(?P<bare_year>[0-9]{4})"
    rf")(?!\w|{_HORIZONTAL_SPACE}*{_SEPARATOR}{_HORIZONTAL_SPACE}*[0-9])",
    re.IGNORECASE,
)
_DECIMAL_PAIR = re.compile(
    rf"[0-9]{{1,3}}{_HORIZONTAL_SPACE}*\.{_HORIZONTAL_SPACE}*[0-9]{{1,3}}"
)
_NAMED_PARTS = re.compile(
    rf"(?:(?P<day>[0-9]+){_HORIZONTAL_SPACE}*)?"
    rf"(?P<month>{_MONTH_PATTERN}){_HORIZONTAL_SPACE}*(?P<year>[0-9]+)?",
    re.IGNORECASE,
)

if not Span.has_extension("date"):
    Span.set_extension("date", default=None)


@clinlp_component(
    name="clinlp_date_linker",
    requires=["doc.spans"],
    assigns=["span._.date"],
)
class DateLinker(Pipe):
    """Link entities to full, partial or ambiguous date candidates on the same line.

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
        """Accept plausible day/month ranges without requiring calendar validity."""
        return any(
            1 <= month <= 12 and 1 <= day <= 31
            for month, day in ((first, second), (second, first))
        )

    @staticmethod
    def _normalize_full(
        year: int, month: int, day: int
    ) -> tuple[str | None, str] | None:
        """Normalize a known order without repairing inconsistent calendar dates."""
        if not (1 <= year <= 9999 and 1 <= month <= 12 and 1 <= day <= 31):
            return None
        try:
            return date(year, month, day).isoformat(), "day"
        except ValueError:
            return None, "ambiguous"

    @classmethod
    def _normalize_year_last(
        cls, first: int, second: int, year_text: str
    ) -> tuple[str | None, str] | None:
        """Retain unknown centuries and competing day/month orders as ambiguous."""
        orientations = {
            (month, day)
            for month, day in ((first, second), (second, first))
            if 1 <= month <= 12 and 1 <= day <= 31
        }
        if not orientations or (len(year_text) == 4 and int(year_text) == 0):
            return None
        if len(year_text) == 2 or len(orientations) > 1:
            return None, "ambiguous"
        month, day = next(iter(orientations))
        return cls._normalize_full(int(year_text), month, day)

    @classmethod
    def _normalize_partial(
        cls, parts: list[str], expression: str
    ) -> tuple[str | None, str] | None:
        """Normalize month/year pairs without interpreting decimal short pairs."""
        first, second = parts
        if len(first) == 4 and len(second) <= 2:
            year, month = int(first), int(second)
        elif len(second) == 4 and len(first) <= 2:
            year, month = int(second), int(first)
        else:
            if (
                max(len(first), len(second)) <= 2
                and "." not in expression
                and cls._valid_short_date(int(first), int(second))
            ):
                return None, "ambiguous"
            return None
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}", "month"
        return None

    @classmethod
    def _normalize_numeric(cls, expression: str) -> tuple[str | None, str] | None:
        """Consume numeric chains whole to prevent malformed-date fragments."""
        parts = re.split(
            rf"{_HORIZONTAL_SPACE}*{_SEPARATOR}{_HORIZONTAL_SPACE}*", expression
        )
        if len(parts) == 2:
            return cls._normalize_partial(parts, expression)
        if len(parts) != 3:
            return None
        first, second, third = parts
        if len(first) == 4 and max(len(second), len(third)) <= 2:
            return cls._normalize_full(int(first), int(second), int(third))
        if max(len(first), len(second)) <= 2 and len(third) in (2, 4):
            return cls._normalize_year_last(int(first), int(second), third)
        return None

    @classmethod
    def _normalize_named(cls, expression: str) -> tuple[str | None, str] | None:
        """Read Dutch month names without inferring a missing year or century."""
        parts = _NAMED_PARTS.fullmatch(expression)
        if parts is None:
            return None
        day_text, year_text = parts["day"], parts["year"]
        if day_text is not None and (len(day_text) > 2 or not 1 <= int(day_text) <= 31):
            return None
        if year_text is not None and len(year_text) not in (2, 4):
            return None
        if year_text is None or len(year_text) == 2:
            return None, "ambiguous"
        month = _MONTH_NUMBERS[parts["month"].rstrip(".").casefold()]
        year = int(year_text)
        if day_text is not None:
            return cls._normalize_full(year, month, int(day_text))
        return (f"{year:04d}-{month:02d}", "month") if year else None

    @classmethod
    def _normalize(cls, match: re.Match) -> tuple[str | None, str] | None:
        """Normalize only information supported by the captured expression."""
        if match["numeric"] is not None:
            return cls._normalize_numeric(match["numeric"])
        if match["named"] is not None or match["named_short"] is not None:
            return cls._normalize_named(match["named"] or match["named_short"])
        year = match["bare_year"]
        return (year, "year") if int(year) else None

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
        matches = [
            match
            for match in _DATE_PATTERN.finditer(text, line_start, line_end)
            if not _DECIMAL_PAIR.fullmatch(match["numeric"] or "")
        ]

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
        Values are ISO dates for day precision, YYYY-MM for month precision,
        YYYY for year precision, and ``None`` for ambiguous expressions. Original
        text is retained, including an attached timestamp when present; clock time
        and timezone do not change the day-level value.
        Character offsets refer to the original ``doc.text`` and are end-exclusive.
        A missing span group is treated as empty.
        """
        for entity in doc.spans.get(self.spans_key, ()):
            entity._.date = self._find_date(entity)

        return doc
