"""Grouped public-API checks for permissive numeric-date component ranges.

Loops intentionally keep collection small; every assertion reports the exact input.
The full-date grid covers 3 years x 14 months x 33 days x 4 formats (5,544).
Year-last inputs consider both component orders before calendar validation.
Short dates cover all 33 x 33 pairs in both supported separators (2,178).
No private parsing helpers or production regexes are used as validation oracles.
"""

from datetime import date
from itertools import product

import pytest
import spacy

from clinlp.ie import DateLinker

FULL_FORMATS = ("/", "-", ".", "iso")
GRID_YEARS = (1900, 2000, 2024)
EDGE_YEARS = (0, 1, 4, 100, 400, 1600, 1700, 1800, 2023, 2100, 2400, 9999)

DATE_EXAMPLES = (
    ("29/02/2000", "2000-02-29", "day"),
    ("29-02-2000", "2000-02-29", "day"),
    ("29.02.2000", "2000-02-29", "day"),
    ("2000-02-29", "2000-02-29", "day"),
    ("2/2000", "2000-02", "month"),
    ("02-2000", "2000-02", "month"),
    ("2000/2", "2000-02", "month"),
    ("2000-02", "2000-02", "month"),
    ("29/2", None, "ambiguous"),
    ("2-29", None, "ambiguous"),
    ("31/04/2020", None, "ambiguous"),
    ("29-02-2019", None, "ambiguous"),
    ("31.11.2016", None, "ambiguous"),
    ("1900-02-29", None, "ambiguous"),
    ("30/02", None, "ambiguous"),
    ("04-31", None, "ambiguous"),
)


@pytest.fixture(scope="module")
def date_nlp():
    nlp = spacy.blank("clinlp")
    linker = nlp.add_pipe("clinlp_date_linker")
    assert isinstance(linker, DateLinker)
    return nlp


def _assert_link(
    nlp,
    text,
    date_text=None,
    value=None,
    precision="day",
    *,
    linker=None,
    entity_text="ENTITY",
):
    doc = nlp.make_doc(text)
    start = text.index(entity_text)
    span = doc.char_span(start, start + len(entity_text), label="finding")
    assert span is not None, repr(text)
    doc.spans["ents"] = [span]
    result = nlp(doc) if linker is None else linker(doc)
    assert result is doc, repr(text)
    assert doc.text == text, repr(text)
    assert len(doc.spans["ents"]) == 1, repr(text)
    entity = doc.spans["ents"][0]
    assert (entity.text, entity.label_, entity.start_char, entity.end_char) == (
        entity_text,
        "finding",
        start,
        start + len(entity_text),
    ), repr(text)
    expected = None
    if date_text is not None:
        date_start = text.index(date_text)
        expected = {
            "value": value,
            "text": date_text,
            "start_char": date_start,
            "end_char": date_start + len(date_text),
            "precision": precision,
            "method": "same_line",
        }
    assert entity._.date == expected, repr(text)
    if expected is not None:
        metadata = entity._.date
        assert text[metadata["start_char"] : metadata["end_char"]] == date_text, repr(
            text
        )


def _calendar_outcome(year, month, day):
    if not (1 <= year <= 9999 and 1 <= month <= 12 and 1 <= day <= 31):
        return None, None
    try:
        return date(year, month, day).isoformat(), "day"
    except ValueError:
        return None, "ambiguous"


def _full_text(year, month, day, date_format, day_width=2, month_width=2):
    if date_format == "iso":
        return f"{year:04d}-{month:02d}-{day:02d}"
    return (
        f"{day:0{day_width}d}{date_format}"
        f"{month:0{month_width}d}{date_format}{year:04d}"
    )


def _full_outcome(year, month, day, *, year_first):
    if year_first:
        return _calendar_outcome(year, month, day)
    if not 1 <= year <= 9999:
        return None, None
    orientations = {
        (candidate_month, candidate_day)
        for candidate_month, candidate_day in ((month, day), (day, month))
        if 1 <= candidate_month <= 12 and 1 <= candidate_day <= 31
    }
    if not orientations:
        return None, None
    if len(orientations) > 1:
        return None, "ambiguous"
    unique_month, unique_day = orientations.pop()
    return _calendar_outcome(year, unique_month, unique_day)


def _assert_full(nlp, date_text, year, month, day, *, year_first):
    value, precision = _full_outcome(year, month, day, year_first=year_first)
    _assert_link(
        nlp,
        f"{date_text}: ENTITY",
        date_text if precision is not None else None,
        value,
        precision,
    )


@pytest.mark.parametrize("date_format", FULL_FORMATS)
@pytest.mark.parametrize("year", GRID_YEARS)
def test_full_calendar_grid(date_nlp, date_format, year):
    # Out-of-range dates cannot fall back to fragments; impossible dates stay whole.
    for month, day in product(range(14), range(33)):
        date_text = _full_text(year, month, day, date_format)
        _assert_full(
            date_nlp, date_text, year, month, day, year_first=date_format == "iso"
        )


@pytest.mark.parametrize("date_format", FULL_FORMATS)
def test_year_extremes_and_century_rules(date_nlp, date_format):
    for year, (month, day) in product(
        EDGE_YEARS, ((1, 1), (2, 28), (2, 29), (2, 30), (4, 31), (12, 31))
    ):
        date_text = _full_text(year, month, day, date_format)
        _assert_full(
            date_nlp, date_text, year, month, day, year_first=date_format == "iso"
        )


@pytest.mark.parametrize("separator", ["/", "-", "."])
def test_full_date_component_widths(date_nlp, separator):
    for (year, month, day), day_width, month_width in product(
        (
            (2000, 2, 1),
            (2023, 2, 9),
            (2000, 2, 29),
            (2023, 2, 29),
            (2000, 0, 1),
            (2000, 1, 0),
        ),
        (1, 2),
        (1, 2),
    ):
        date_text = _full_text(year, month, day, separator, day_width, month_width)
        _assert_full(date_nlp, date_text, year, month, day, year_first=False)


def _possible_short_pair(first, second):
    # Only component ranges matter; neither orientation nor month length wins.
    return any(
        1 <= month <= 12 and 1 <= day <= 31
        for month, day in ((first, second), (second, first))
    )


@pytest.mark.parametrize("separator", ["/", "-"])
def test_all_short_pairs(date_nlp, separator):
    for first, second in product(range(33), repeat=2):
        date_text = f"{first}{separator}{second}"
        _assert_link(
            date_nlp,
            f"ENTITY ({date_text}).",
            date_text if _possible_short_pair(first, second) else None,
            precision="ambiguous",
        )


@pytest.mark.parametrize("separator", ["/", "-"])
def test_short_pair_component_widths(date_nlp, separator):
    for (first, second), first_width, second_width in product(
        ((2, 9), (2, 29), (29, 2), (2, 30), (0, 9), (9, 0)), (1, 2), (1, 2)
    ):
        date_text = f"{first:0{first_width}d}{separator}{second:0{second_width}d}"
        _assert_link(
            date_nlp,
            f"{date_text} ENTITY",
            date_text if _possible_short_pair(first, second) else None,
            precision="ambiguous",
        )


@pytest.mark.parametrize("separator", ["/", "-"])
@pytest.mark.parametrize("year_first", [False, True])
def test_month_year_grid(date_nlp, separator, year_first):
    for year, month, width in product(
        (0, 1, 1900, 2000, 2024, 9999), range(14), (1, 2)
    ):
        components = [f"{month:0{width}d}", f"{year:04d}"]
        if year_first:
            components.reverse()
        date_text = separator.join(components)
        _, precision = _calendar_outcome(year, month, 1)
        valid = precision == "day"
        _assert_link(
            date_nlp,
            f"ENTITY: {date_text}",
            date_text if valid else None,
            f"{year:04d}-{month:02d}" if valid else None,
            "month",
        )


@pytest.mark.parametrize(("date_text", "value", "precision"), DATE_EXAMPLES)
def test_punctuation_and_absolute_offsets(date_nlp, date_text, value, precision):
    for template in (
        "{date} ENTITY",
        "ENTITY {date}",
        "({date}): ENTITY",
        "ENTITY ({date}).",
        "[{date}]; ENTITY",
        "ENTITY [{date}]!",
        '"{date}", ENTITY',
        'ENTITY: "{date}"?',
        "één: {date}\tENTITY",
        "één: ENTITY\t{date}.",
    ):
        _assert_link(
            date_nlp, template.format(date=date_text), date_text, value, precision
        )


@pytest.mark.parametrize("linebreak", ["\r", "\n", "\r\n"])
def test_physical_lines_never_inherit_dates(date_nlp, linebreak):
    for date_text, value, precision in DATE_EXAMPLES:
        for text in (
            f"{date_text}{linebreak}ENTITY",
            f"ENTITY{linebreak}{date_text}",
            f"{date_text}{linebreak}{linebreak}ENTITY",
            f"{date_text}{linebreak}ENTITY{linebreak}{date_text}",
        ):
            _assert_link(date_nlp, text)
        for text in (
            f"01/01/1999{linebreak}ENTITY {date_text}",
            f"{date_text} ENTITY{linebreak}01/01/1999",
        ):
            _assert_link(date_nlp, text, date_text, value, precision)


@pytest.mark.parametrize("linebreak", ["\r", "\n", "\r\n"])
def test_entity_crossing_physical_lines_is_unresolved(date_nlp, linebreak):
    for date_text, _, _ in DATE_EXAMPLES:
        entity_text = f"ENTITY{linebreak}CONTINUED"
        _assert_link(date_nlp, f"{date_text} {entity_text}", entity_text=entity_text)


@pytest.mark.parametrize("date_first", [False, True])
def test_inclusive_distances_for_every_precision(date_nlp, date_first):
    for max_distance, gap in (
        (0, 1),
        (1, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (None, 199),
        (None, 200),
        (None, 201),
        (250, 201),
    ):
        linker = None if max_distance is None else DateLinker(max_distance=max_distance)
        limit = 200 if max_distance is None else max_distance
        for date_text, value, precision in DATE_EXAMPLES:
            parts = [date_text, "ENTITY"] if date_first else ["ENTITY", date_text]
            text = (" " * gap).join(parts)
            _assert_link(
                date_nlp,
                text,
                date_text if gap <= limit else None,
                value,
                precision,
                linker=linker,
            )


@pytest.mark.parametrize("date_first", [False, True])
def test_zero_gap_with_touching_nonoverlapping_span(date_nlp, date_first):
    linker = DateLinker(max_distance=0)
    for date_text, value, precision in DATE_EXAMPLES:
        # Include the punctuation in the entity to make the character gap zero
        # without joining the date to an identifier (which is unsupported).
        text = f"{date_text}(ENTITY)" if date_first else f"(ENTITY){date_text}"
        _assert_link(
            date_nlp,
            text,
            date_text,
            value,
            precision,
            linker=linker,
            entity_text="(ENTITY)",
        )


@pytest.mark.parametrize("date_text", [example[0] for example in DATE_EXAMPLES])
def test_overlapping_dates_are_not_linked(date_nlp, date_text):
    _assert_link(date_nlp, f"ENTITY ({date_text})", entity_text=f"ENTITY ({date_text})")


@pytest.mark.parametrize("date_text", [example[0] for example in DATE_EXAMPLES])
def test_multiple_expressions_remain_unresolved(date_nlp, date_text):
    for text in (
        f"{date_text} ENTITY {date_text}",
        f"{date_text} ENTITY 31/02/2000",
        f"{date_text} ENTITY" + " " * 201 + "1/2000",
    ):
        _assert_link(date_nlp, text)


@pytest.mark.parametrize("date_text", [example[0] for example in DATE_EXAMPLES])
def test_embedded_identifiers_and_long_sequences(date_nlp, date_text):
    completed_dates = {
        f"1{separator}{partial}": (None, "ambiguous")
        for separator, partial in product(
            ("/", "-", "."), ("2/2000", "02-2000", "2-29", "30/02", "04-31")
        )
    }
    completed_dates.update(
        {
            f"{partial}{separator}1": ("2000-02-01", "day")
            for separator, partial in product(("/", "-", "."), ("2000/2", "2000-02"))
        }
    )
    for wrapper in (
        "id_{}",
        "{}_id",
        "abc{}",
        "{}abc",
        "é{}",
        "{}é",
        "123{}",
        "{}123",
        "1/{}",
        "{}/1",
        "1-{}",
        "{}-1",
        "1.{}",
        "{}.1",
    ):
        wrapped = wrapper.format(date_text)
        # Completing a partial date is valid, including mixed separators.
        if wrapped in completed_dates:
            value, precision = completed_dates[wrapped]
            _assert_link(date_nlp, f"{wrapped} ENTITY", wrapped, value, precision)
        else:
            _assert_link(date_nlp, f"{wrapped} ENTITY")


@pytest.mark.parametrize(
    "expressions",
    [
        (
            "09-10-016",
            "1/2020/9",
            "20/05/016",
            "2016-005-2",
            "120/05/2016",
            "20/005/2016",
            "20/05/20160",
            "12016-05-20",
        ),
        (
            "2016-005-20",
            "2016-05-200",
            "12020-9",
            "2020-999",
            "109/10",
            "09/100",
            "00000/02",
            "02/00000",
            "1/2/2000/3",
            "2000-02-29-1",
            "2000/2/29/1",
            "29.02.2000.1",
        ),
    ],
    ids=["malformed-and-overlong-full", "overlong-partial-and-extra-components"],
)
def test_malformed_expressions_have_no_fragment_fallback(date_nlp, expressions):
    for date_text in expressions:
        _assert_link(date_nlp, f"ENTITY ({date_text})")


@pytest.mark.parametrize(
    "expressions",
    [
        ("February", "mei", "gisteren", "vandaag", "morgen"),
        ("0000", "000", "201", "10000", "20/05/016"),
        ("9.10", "2020.009", "2016-005-20", "2016-05-002", "2016/05/200"),
    ],
    ids=["bare-names-and-relative", "invalid-years", "unsupported-numeric-shapes"],
)
def test_existing_unsupported_formats_remain_unlinked(date_nlp, expressions):
    for date_text in expressions:
        _assert_link(date_nlp, f"{date_text} ENTITY")
