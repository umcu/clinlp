"""High-recall date formats through the public linker, with small collection cost."""

from itertools import product

import pytest
import spacy
from spacy.tokens import Doc

from clinlp.ie import DateLinker

SEPARATORS = (
    "/",
    ".",
    "-",
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2015",
    "\u2212",
)
MONTH_NAMES = (
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
)
USER_EXAMPLES = (
    ("20 mei 2020", "2020-05-20", "day"),
    ("2020-9-1", "2020-09-01", "day"),
    ("2020/09/01", "2020-09-01", "day"),
    ("09/20/2020", "2020-09-20", "day"),
    ("20/09/20", None, "ambiguous"),
    ("2020", "2020", "year"),
    ("20 / 09 / 2020", "2020-09-20", "day"),
    ("2020\u20139", "2020-09", "month"),
    ("2020-09-01T12:00", "2020-09-01", "day"),
)


@pytest.fixture(scope="module")
def date_nlp():
    nlp = spacy.blank("clinlp")
    assert isinstance(nlp.add_pipe("clinlp_date_linker"), DateLinker)
    return nlp


def _link(nlp, text, *, linker=None):
    doc = nlp.make_doc(text)
    start = text.index("ENTITY")
    span = doc.char_span(start, start + len("ENTITY"), label="finding")
    assert span is not None, repr(text)
    doc.spans["ents"] = [span]
    result = nlp(doc) if linker is None else linker(doc)
    assert result is doc, repr(text)
    assert doc.text == text, repr(text)
    assert list(doc.spans["ents"]) == [span], repr(text)
    return doc


def _assert_date(
    nlp, raw, value, precision, *, template="één: ENTITY [{date}]!", linker=None
):
    text = template.format(date=raw)
    doc = _link(nlp, text, linker=linker)
    start = text.index(raw)
    metadata = doc.spans["ents"][0]._.date
    assert metadata == {
        "value": value,
        "text": raw,
        "start_char": start,
        "end_char": start + len(raw),
        "precision": precision,
        "method": "same_line",
    }, repr(text)
    assert text[metadata["start_char"] : metadata["end_char"]] == raw, repr(text)
    return doc


def _assert_unlinked(nlp, raw):
    text = f"één: ENTITY [{raw}]!"
    doc = _link(nlp, text)
    assert doc.spans["ents"][0]._.date is None, repr(text)


def test_decimal_doses_do_not_suppress_dates(date_nlp):
    for number in ("2.5", "9.10", "123.456", "2 . 5"):
        _assert_date(
            date_nlp,
            "20/09/2020",
            "2020-09-20",
            "day",
            template="{date} ENTITY dosis " + number + " mg",
        )


def test_all_user_examples(date_nlp):
    for raw, value, precision in USER_EXAMPLES:
        for template in ("één: {date}\tENTITY", "één: ENTITY [{date}]!"):
            _assert_date(date_nlp, raw, value, precision, template=template)


def test_full_numeric_separators_and_order(date_nlp):
    # Mixed separators are as valid as matching separators in every orientation.
    for first_separator, second_separator in product(SEPARATORS, repeat=2):
        for first, second, third, value, precision in (
            ("2020", "9", "1", "2020-09-01", "day"),
            ("20", "9", "2020", "2020-09-20", "day"),
            ("9", "20", "2020", "2020-09-20", "day"),
            ("03", "04", "2016", None, "ambiguous"),
            ("05", "05", "2016", "2016-05-05", "day"),
        ):
            raw = f"{first}{first_separator}{second}{second_separator}{third}"
            _assert_date(date_nlp, raw, value, precision)


def test_year_first_component_widths(date_nlp):
    for separator, month, day in product(SEPARATORS, ("9", "09"), ("1", "01")):
        _assert_date(
            date_nlp, f"2020{separator}{month}{separator}{day}", "2020-09-01", "day"
        )


def test_previously_unsupported_numeric_formats(date_nlp):
    for raw, value, precision in (
        ("2020-9/20", "2020-09-20", "day"),
        ("09/10/16", None, "ambiguous"),
        ("09/10.1234", None, "ambiguous"),
        ("2016-5-20", "2016-05-20", "day"),
        ("2016-05-2", "2016-05-02", "day"),
        ("2016/05/20", "2016-05-20", "day"),
        ("2016.05.20", "2016-05-20", "day"),
        ("20/05-2016", "2016-05-20", "day"),
        ("20-05.2016", "2016-05-20", "day"),
        ("20.05/2016", "2016-05-20", "day"),
        ("2016-05/20", "2016-05-20", "day"),
        ("05/20/2016", "2016-05-20", "day"),
        ("2000/2/29", "2000-02-29", "day"),
    ):
        _assert_date(date_nlp, raw, value, precision)


def test_horizontal_whitespace_preserves_exact_offsets(date_nlp):
    for separator, whitespace in product(
        SEPARATORS, (" ", "\t", "\u00a0", "\u2003", "\u202f")
    ):
        for raw, value, precision in (
            (
                f"2020{whitespace}{separator}{whitespace}9{separator}{whitespace}1",
                "2020-09-01",
                "day",
            ),
            (
                f"20{whitespace}{separator}09{whitespace}{separator}{whitespace}2020",
                "2020-09-20",
                "day",
            ),
            (f"2020{whitespace}{separator}{whitespace}9", "2020-09", "month"),
        ):
            _assert_date(date_nlp, raw, value, precision)


def test_partial_numeric_dates(date_nlp):
    for separator in SEPARATORS:
        for raw in (f"2020{separator}9", f"09{separator}2020"):
            _assert_date(date_nlp, raw, "2020-09", "month")
        for first, second in ((9, 10), (20, 9), (9, 20), (31, 4), (4, 31)):
            raw = f"{first}{separator}{second}"
            if separator == ".":
                _assert_unlinked(date_nlp, raw)
            else:
                _assert_date(date_nlp, raw, None, "ambiguous")
        for raw in (
            f"2020{separator}13",
            f"0{separator}2020",
            f"0000{separator}9",
            f"13{separator}14",
            f"0{separator}9",
        ):
            _assert_unlinked(date_nlp, raw)


def test_two_digit_years_never_guess_a_century(date_nlp):
    # All 100 spellings include 00: its unknown century must not become year zero.
    for year in range(100):
        for first, second in ((20, 9), (9, 20), (3, 4), (5, 5), (29, 2), (31, 4)):
            _assert_date(date_nlp, f"{first}/{second}/{year:02d}", None, "ambiguous")
    for separator in SEPARATORS:
        _assert_date(date_nlp, f"20{separator}05{separator}16", None, "ambiguous")
        for first, second in ((0, 9), (9, 0), (32, 9), (9, 32), (13, 14)):
            _assert_unlinked(date_nlp, f"{first}{separator}{second}{separator}00")


def test_bare_four_digit_years(date_nlp):
    for year in (1, 4, 99, 100, 999, 1000, 1900, 2016, 2020, 9999):
        raw = f"{year:04d}"
        _assert_date(date_nlp, raw, raw, "year")
    for raw in ("0000", "000", "20", "202", "10000", "02020"):
        _assert_unlinked(date_nlp, raw)


def test_unsupported_english_month_does_not_hide_bare_year(date_nlp):
    _assert_unlinked(date_nlp, "February")
    for template in ("February {date} ENTITY", "ENTITY February {date}"):
        _assert_date(date_nlp, "2000", "2000", "year", template=template)


def test_dutch_month_names_and_aliases(date_nlp):
    for month, names in enumerate(MONTH_NAMES, start=1):
        for name in names:
            variants = (name, name.upper(), name.title())
            if name != names[0]:
                variants += (f"{name}.", f"{name.upper()}.")
            for variant in variants:
                _assert_date(
                    date_nlp, f"20 {variant} 2020", f"2020-{month:02d}-20", "day"
                )
                _assert_date(date_nlp, f"{variant} 2020", f"2020-{month:02d}", "month")
    for raw, value, precision in (
        ("20 mei 2016", "2016-05-20", "day"),
        ("mei 2016", "2016-05", "month"),
        ("20mei2020", "2020-05-20", "day"),
        ("mei2020", "2020-05", "month"),
        ("1 mei 2020", "2020-05-01", "day"),
        ("01mei2020", "2020-05-01", "day"),
        ("20mei0001", "0001-05-20", "day"),
        ("mei0001", "0001-05", "month"),
        ("20\tMEI\t2020", "2020-05-20", "day"),
        ("20\u00a0mei\u00a02020", "2020-05-20", "day"),
        ("20mei", None, "ambiguous"),
        ("20 mei", None, "ambiguous"),
        ("20 mei 20", None, "ambiguous"),
        ("20mei00", None, "ambiguous"),
        ("mei 20", None, "ambiguous"),
        ("mei00", None, "ambiguous"),
    ):
        _assert_date(date_nlp, raw, value, precision)


def test_named_dates_validate_explicit_components(date_nlp):
    for raw in (
        "0 mei 2020",
        "00mei2020",
        "32 mei 2020",
        "20 mei 0000",
        "mei 0000",
        "20 mei 020",
        "mei 020",
        "120 mei 2020",
        "0mei",
        "32mei",
        "February",
    ):
        _assert_unlinked(date_nlp, raw)
    for names in MONTH_NAMES:
        for name in names:
            _assert_unlinked(date_nlp, name)
    for raw, value, precision in (
        ("29 februari 2000", "2000-02-29", "day"),
        ("29 februari 1900", None, "ambiguous"),
        ("31 april 2020", None, "ambiguous"),
        ("30feb2020", None, "ambiguous"),
        ("29 feb 00", None, "ambiguous"),
    ):
        _assert_date(date_nlp, raw, value, precision)


def test_calendar_inconsistency_is_captured_in_both_orders(date_nlp):
    for separator in SEPARATORS:
        for components in (
            ("31", "04", "2020"),
            ("04", "31", "2020"),
            ("2020", "04", "31"),
            ("29", "02", "1900"),
            ("02", "29", "1900"),
        ):
            _assert_date(date_nlp, separator.join(components), None, "ambiguous")
        for components in (
            ("29", "02", "2000"),
            ("02", "29", "2000"),
            ("2000", "02", "29"),
        ):
            _assert_date(date_nlp, separator.join(components), "2000-02-29", "day")


def test_timestamps_capture_suffix_without_time_interpretation(date_nlp):
    for marker, suffix in product(
        ("T", "t"),
        (
            "12:00",
            "12:00:59",
            "12:00:59.123",
            "12:00Z",
            "12:00+02:00",
            "12:00+0200",
            "12:00:59.123Z",
            "12:00:59.123+02:00",
            "12:00:59.123+0200",
            "25:61:99",
        ),
    ):
        raw = f"2020-09-01{marker}{suffix}"
        _assert_date(date_nlp, raw, "2020-09-01", "day")
    for raw, value, precision in (
        ("2020/9/1T12:00", "2020-09-01", "day"),
        ("09/20/2020t12:00", "2020-09-20", "day"),
        ("03/04/2016T12:00", None, "ambiguous"),
        ("20/09/20T12:00", None, "ambiguous"),
        ("2020-04-31T12:00", None, "ambiguous"),
    ):
        _assert_date(date_nlp, raw, value, precision)


def test_invalid_numeric_expressions_never_fall_back_to_fragments(date_nlp):
    for separator in SEPARATORS:
        for components in (
            ("0000", "9", "1"),
            ("2020", "0", "1"),
            ("2020", "13", "1"),
            ("2020", "9", "0"),
            ("2020", "9", "32"),
            ("0", "9", "2020"),
            ("9", "0", "2020"),
            ("32", "9", "2020"),
            ("13", "14", "2020"),
            ("20", "9", "0000"),
            ("2020", "009", "1"),
            ("2020", "9", "001"),
            ("20", "009", "2020"),
            ("20", "9", "020"),
            ("1", "2020", "9"),
            ("1", "2", "2020", "3"),
        ):
            _assert_unlinked(date_nlp, separator.join(components))
        _assert_unlinked(date_nlp, f"1 {separator} 2 / 2020 - 3")


def test_new_formats_do_not_match_inside_identifiers(date_nlp):
    for raw in (
        "2020",
        "20mei2020",
        "mei2020",
        "20mei",
        "2020\u20139",
        "2020-09-01T12:00",
    ):
        for wrapper in (
            "id_{}",
            "{}_id",
            "abc{}",
            "{}abc",
            "é{}",
            "{}é",
            "123{}",
            "{}123",
        ):
            _assert_unlinked(date_nlp, wrapper.format(raw))


def test_whole_dates_are_one_candidate_but_separate_year_is_another(date_nlp):
    for raw, value, precision in USER_EXAMPLES:
        _assert_date(date_nlp, raw, value, precision)
        for text in (
            f"{raw} ENTITY 1999",
            f"1999 ENTITY {raw}",
            f"{raw} ENTITY" + " " * 201 + "1999",
        ):
            doc = _link(date_nlp, text)
            assert doc.spans["ents"][0]._.date is None, repr(text)


def test_new_formats_keep_same_line_and_distance_rules(date_nlp):
    linker = DateLinker(max_distance=3)
    for raw, value, precision in USER_EXAMPLES:
        for template in ("{date}   ENTITY", "ENTITY   {date}"):
            _assert_date(
                date_nlp, raw, value, precision, template=template, linker=linker
            )
        for text in (f"{raw}    ENTITY", f"ENTITY    {raw}"):
            doc = _link(date_nlp, text, linker=linker)
            assert doc.spans["ents"][0]._.date is None, repr(text)
        for linebreak in ("\n", "\r", "\r\n"):
            for text in (f"{raw}{linebreak}ENTITY", f"ENTITY{linebreak}{raw}"):
                doc = _link(date_nlp, text)
                assert doc.spans["ents"][0]._.date is None, repr(text)


def test_date_components_cannot_consume_line_separators(date_nlp):
    for linebreak in ("\r", "\n", "\r\n", "\u2028", "\u2029"):
        for raw in (
            f"2020 /{linebreak}09 / 20",
            f"20 / 09{linebreak}/ 2020",
            f"20{linebreak}mei 2020",
            f"20 mei{linebreak}2020",
        ):
            doc = _link(date_nlp, f"{raw} ENTITY")
            metadata = doc.spans["ents"][0]._.date
            # A fragment wholly on one line may be a date in its own right;
            # whitespace must never turn the entire broken expression into one.
            if metadata is not None:
                assert metadata["text"] != raw, repr(raw)
                assert not any(
                    char in metadata["text"] for char in "\r\n\u2028\u2029"
                ), repr(raw)


def test_new_metadata_survives_doc_serialization(date_nlp):
    for raw, value, precision in USER_EXAMPLES:
        doc = _assert_date(date_nlp, raw, value, precision)
        restored = Doc(date_nlp.vocab).from_bytes(doc.to_bytes())
        assert restored.text == doc.text, repr(raw)
        assert restored.spans["ents"][0]._.date == doc.spans["ents"][0]._.date, repr(
            raw
        )
