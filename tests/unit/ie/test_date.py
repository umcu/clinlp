import pytest
import spacy
from spacy.tokens import Doc, Span

from clinlp.ie import DateLinker
from clinlp.ie.qualifier import Qualifier, get_qualifiers, set_qualifiers


def expected_date(text, value, start_char=0, precision="day"):
    return {
        "value": value,
        "text": text,
        "start_char": start_char,
        "end_char": start_char + len(text),
        "precision": precision,
        "method": "same_line",
    }


class TestDateLinker:
    @pytest.mark.parametrize(
        ("date_text", "value", "precision"),
        [
            ("2020-9", "2020-09", "month"),
            ("2020/9", "2020-09", "month"),
            ("2020-09", "2020-09", "month"),
            ("2020/09", "2020-09", "month"),
            ("9/2020", "2020-09", "month"),
            ("09-2020", "2020-09", "month"),
            ("2016-05", "2016-05", "month"),
            ("05/2016", "2016-05", "month"),
            ("09/10", None, "ambiguous"),
            ("9-10", None, "ambiguous"),
            ("20/09", None, "ambiguous"),
            ("09/20", None, "ambiguous"),
            ("20-9", None, "ambiguous"),
            ("9-20", None, "ambiguous"),
            ("20/05", None, "ambiguous"),
            ("29/02", None, "ambiguous"),
            ("2-29", None, "ambiguous"),
            ("12/12", None, "ambiguous"),
            ("31/04", None, "ambiguous"),
            ("04-31", None, "ambiguous"),
            ("30/02", None, "ambiguous"),
            ("02-30", None, "ambiguous"),
        ],
    )
    @pytest.mark.parametrize("template", ["{date}:..ENTITY", "ENTITY ({date})."])
    def test_partial_dates(self, nlp_entity, date_text, value, precision, template):
        text = template.format(date=date_text)
        nlp_entity.add_pipe("clinlp_date_linker")
        doc = nlp_entity(text)
        assert doc.spans["ents"][0]._.date == expected_date(
            date_text, value, text.index(date_text), precision
        )

    @pytest.mark.parametrize(
        "date_text",
        [
            "2020-0",
            "2020/13",
            "0000-9",
            "13/2020",
            "00-2020",
            "13/14",
            "00/09",
            "9-0",
            "32-1",
            "1/32",
            "9.10",
            "abc2020-9",
            "2020-9x",
            "id_09/10",
            "09/10_id",
            "09-10-016",
            "12020-9",
            "2020-999",
            "109/10",
            "09/100",
            "1/2020/9",
        ],
    )
    def test_invalid_or_embedded_partial_dates(self, nlp_entity, date_text):
        doc = nlp_entity(f"{date_text} ENTITY")
        DateLinker()(doc)
        assert doc.spans["ents"][0]._.date is None

    @pytest.mark.parametrize(
        "text",
        [
            "2020-9 ENTITY 09/10",
            "20/05/2016 ENTITY 09/10",
            "09/10 ENTITY 31/04",
            "2020-13 ENTITY 2020-9",
        ],
    )
    def test_partial_dates_also_make_multiple_expression_lines_unresolved(
        self, nlp_entity, text
    ):
        doc = nlp_entity(text)
        DateLinker()(doc)
        assert doc.spans["ents"][0]._.date is None

    def test_partial_date_serialization_and_pipe(self, nlp_entity):
        nlp_entity.add_pipe("clinlp_date_linker")
        texts = [
            "2020-9: ENTITY",
            "09/10 ENTITY",
            "ENTITY",
            "31/02/2016 ENTITY",
            "30/02 ENTITY",
        ]
        docs = list(nlp_entity.pipe(texts))
        metadata = [doc.spans["ents"][0]._.date for doc in docs]
        assert metadata == [
            expected_date("2020-9", "2020-09", precision="month"),
            expected_date("09/10", None, precision="ambiguous"),
            None,
            expected_date("31/02/2016", None, precision="ambiguous"),
            expected_date("30/02", None, precision="ambiguous"),
        ]
        for doc, expected in zip(docs, metadata, strict=True):
            restored = Doc(nlp_entity.vocab).from_bytes(doc.to_bytes())
            assert restored.spans["ents"][0]._.date == expected

    def test_factory(self, nlp_entity):
        # Act
        linker = nlp_entity.add_pipe(
            "clinlp_date_linker", config={"spans_key": "ents", "max_distance": 200}
        )
        doc = nlp_entity("20/05/2016 ENTITY")

        # Assert
        assert isinstance(linker, DateLinker)
        date = doc.spans["ents"][0]._.date
        assert type(date) is dict
        assert date == expected_date("20/05/2016", "2016-05-20")

    @pytest.mark.parametrize("max_distance", [-1, -200])
    def test_negative_max_distance(self, max_distance):
        # Act & Assert
        # The API specifies the exception type, not the error message.
        with pytest.raises(ValueError):  # noqa: PT011
            DateLinker(max_distance=max_distance)

    @pytest.mark.parametrize(
        ("date_text", "value", "precision"),
        [
            ("20/05/2016", "2016-05-20", "day"),
            ("20-05-2016", "2016-05-20", "day"),
            ("20.05.2016", "2016-05-20", "day"),
            ("1/2/2016", None, "ambiguous"),
            ("01/2/2016", None, "ambiguous"),
            ("1/02/2016", None, "ambiguous"),
            ("1-2-2016", None, "ambiguous"),
            ("01-2-2016", None, "ambiguous"),
            ("1-02-2016", None, "ambiguous"),
            ("1.2.2016", None, "ambiguous"),
            ("01.2.2016", None, "ambiguous"),
            ("1.02.2016", None, "ambiguous"),
            ("03/04/2016", None, "ambiguous"),
            ("2016-05-20", "2016-05-20", "day"),
            ("29/02/2016", "2016-02-29", "day"),
            ("29.02.2000", "2000-02-29", "day"),
            ("2000-02-29", "2000-02-29", "day"),
            ("31/12/2016", "2016-12-31", "day"),
        ],
    )
    @pytest.mark.parametrize("template", ["{date} ENTITY", "ENTITY ({date})."])
    def test_explicit_dates(self, nlp_entity, date_text, value, precision, template):
        # Arrange
        text = template.format(date=date_text)
        doc = nlp_entity(text)

        # Act
        result = DateLinker()(doc)

        # Assert
        assert result is doc
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date == expected_date(
            date_text, value, text.index(date_text), precision
        )

    @pytest.mark.parametrize(
        "date_text",
        [
            "31/04/2016",
            "31-06-2016",
            "31.11.2016",
            "29/02/2015",
            "29/02/1900",
            "30/02/2016",
            "2016-04-31",
            "2015-02-29",
            "1900-02-29",
            "31/04/2020",
            "29/02/2019",
            "31/02/2016",
        ],
    )
    @pytest.mark.parametrize("template", ["{date} ENTITY", "ENTITY ({date})."])
    def test_impossible_full_dates_retain_raw_metadata(
        self, nlp_entity, date_text, template
    ):
        text = template.format(date=date_text)
        doc = nlp_entity(text)

        assert DateLinker()(doc) is doc
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date == expected_date(
            date_text, None, text.index(date_text), "ambiguous"
        )

    @pytest.mark.parametrize(
        "date_text",
        [
            "",
            "gisteren",
            "vandaag",
            "morgen",
            "vorige week",
            "February",
            "mei",
            "20/05/016",
            "00/05/2016",
            "20/00/2016",
            "32/05/2016",
            "20/13/2016",
            "20/05/0000",
            "2016-00-20",
            "2016-05-00",
            "2016-13-20",
            "0000-05-20",
        ],
    )
    def test_unsupported_or_invalid_dates(self, nlp_entity, date_text):
        # Arrange
        doc = nlp_entity(f"{date_text} ENTITY")

        # Act
        DateLinker()(doc)

        # Assert
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date is None

    @pytest.mark.parametrize(
        "date_text",
        [
            "abc20/05/2016",
            "20/05/2016abc",
            "id_20/05/2016",
            "20/05/2016_id",
            "abc2016-05-20",
            "2016-05-20abc",
            "id_2016-05-20",
            "2016-05-20_id",
            "120/05/2016",
            "20/005/2016",
            "20/05/20160",
            "12016-05-20",
            "2016-05-200",
            "1/20/05/2016",
            "20/05/2016/1",
            "1-20-05-2016",
            "20-05-2016-1",
            "1.20.05.2016",
            "20.05.2016.1",
            "1-2016-05-20",
            "2016-05-20-1",
        ],
    )
    def test_embedded_dates_are_not_substring_matches(self, nlp_entity, date_text):
        # Arrange
        doc = nlp_entity(f"{date_text} ENTITY")

        # Act
        DateLinker()(doc)

        # Assert
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date is None

    @pytest.mark.parametrize(
        "text",
        [
            "20/05/2016 ENTITY 21/05/2016",
            "20/05/2016 21/05/2016 ENTITY",
            "ENTITY 20/05/2016 21/05/2016",
            "20/05/2016 ENTITY 20/05/2016",
            "20/05/2016 ENTITY 2016-05-20",
            "20/05/2016 ENTITY 31/02/2016",
            "31/02/2016 ENTITY 20/05/2016",
            "2016-02-31 ENTITY 2016-05-20",
            "31/02/2016 ENTITY 32/02/2016",
            "20/05/2016 ENTITY" + " " * 201 + "21/05/2016",
            "20/05/2016 ENTITY" + " " * 201 + "31/02/2016",
        ],
    )
    def test_multiple_date_expressions_are_ambiguous(self, nlp_entity, text):
        # Arrange
        doc = nlp_entity(text)

        # Act
        DateLinker()(doc)

        # Assert
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date is None

    @pytest.mark.parametrize("linebreak", ["\n", "\r", "\r\n"])
    @pytest.mark.parametrize(
        "template",
        [
            "20/05/2016{linebreak}ENTITY",
            "ENTITY{linebreak}20/05/2016",
            "20/05/2016{linebreak}{linebreak}ENTITY",
            "20/05/2016{linebreak}ENTITY{linebreak}21/05/2016",
        ],
    )
    def test_no_cross_line_fallback(self, nlp_entity, linebreak, template):
        # Arrange
        doc = nlp_entity(template.format(linebreak=linebreak))

        # Act
        DateLinker()(doc)

        # Assert
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date is None

    @pytest.mark.parametrize("linebreak", ["\n", "\r", "\r\n"])
    def test_repeated_concepts_on_different_rows(self, nlp, linebreak):
        # Arrange
        ruler = nlp.add_pipe("span_ruler", config={"spans_key": "ents"})
        ruler.add_patterns(
            [
                {"label": "A", "pattern": "CONCEPT A"},
                {"label": "B", "pattern": "CONCEPT B"},
            ]
        )
        nlp.add_pipe("clinlp_date_linker")
        rows = [
            "20/05/2016 CONCEPT A",
            "21/06/2017 CONCEPT A",
            "CONCEPT B 2018-07-22",
            "CONCEPT A",
        ]
        text = linebreak.join(rows)

        # Act
        doc = nlp(text)

        # Assert
        assert doc.text == text
        assert [(span.text, span.label_) for span in doc.spans["ents"]] == [
            ("CONCEPT A", "A"),
            ("CONCEPT A", "A"),
            ("CONCEPT B", "B"),
            ("CONCEPT A", "A"),
        ]
        assert [span._.date for span in doc.spans["ents"]] == [
            expected_date("20/05/2016", "2016-05-20"),
            expected_date("21/06/2017", "2017-06-21", text.index("21/06/2017")),
            expected_date("2018-07-22", "2018-07-22", text.index("2018-07-22")),
            None,
        ]

    @pytest.mark.parametrize("linebreak", ["\n", "\r", "\r\n"])
    def test_span_crossing_linebreak_is_not_linked(self, nlp, linebreak):
        # Arrange
        text = f"20/05/2016 CONCEPT{linebreak}A"
        doc = nlp(text)
        span = doc.char_span(text.index("CONCEPT"), len(text), label="A")
        assert span is not None
        doc.spans["ents"] = [span]

        # Act
        DateLinker()(doc)

        # Assert
        assert doc.spans["ents"][0]._.date is None

    @pytest.mark.parametrize("bounds", [(0, 1), (0, 2), (1, 2), (0, 3)])
    def test_date_overlapping_span_is_not_linked(self, nlp, bounds):
        # Arrange
        doc = Doc(
            nlp.vocab,
            words=["20/05/", "2016", "ENTITY"],
            spaces=[False, True, False],
        )
        doc.spans["ents"] = [Span(doc, *bounds, label="named_entity")]

        # Act
        DateLinker()(doc)

        # Assert
        assert doc.spans["ents"][0]._.date is None

    @pytest.mark.parametrize("gap", [1, 199, 200, 201])
    @pytest.mark.parametrize("date_first", [True, False])
    def test_default_distance_in_characters(self, nlp_entity, gap, date_first):
        # Arrange
        parts = ["20/05/2016", "ENTITY"]
        if not date_first:
            parts.reverse()
        text = (" " * gap).join(parts)
        doc = nlp_entity(text)

        # Act
        DateLinker()(doc)

        # Assert
        expected = (
            expected_date("20/05/2016", "2016-05-20", text.index("20/05/2016"))
            if gap <= 200
            else None
        )
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date == expected

    @pytest.mark.parametrize(
        ("max_distance", "gap", "linked"),
        [(0, 1, False), (3, 2, True), (3, 3, True), (3, 4, False), (250, 201, True)],
    )
    def test_configurable_distance(self, nlp_entity, max_distance, gap, linked):
        # Arrange
        nlp_entity.add_pipe("clinlp_date_linker", config={"max_distance": max_distance})
        text = "20/05/2016" + " " * gap + "ENTITY"

        # Act
        doc = nlp_entity(text)

        # Assert
        expected = expected_date("20/05/2016", "2016-05-20") if linked else None
        assert len(doc.spans["ents"]) == 1
        assert doc.spans["ents"][0]._.date == expected

    @pytest.mark.parametrize("date_first", [True, False])
    def test_zero_distance_allows_touching_nonoverlapping_span(self, nlp, date_first):
        # Arrange
        words = ["20/05/2016", "(ENTITY)"]
        if not date_first:
            words.reverse()
        doc = Doc(nlp.vocab, words=words, spaces=[False, False])
        index = words.index("(ENTITY)")
        doc.spans["ents"] = [Span(doc, index, index + 1, label="named_entity")]

        # Act
        DateLinker(max_distance=0)(doc)

        # Assert
        assert doc.spans["ents"][0]._.date == expected_date(
            "20/05/2016", "2016-05-20", doc.text.index("20/05/2016")
        )

    def test_configurable_span_group(self, nlp):
        # Arrange
        ruler = nlp.add_pipe("span_ruler", config={"spans_key": "findings"})
        ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])
        nlp.add_pipe("clinlp_date_linker", config={"spans_key": "findings"})

        # Act
        doc = nlp("20/05/2016 ENTITY")

        # Assert
        assert "ents" not in doc.spans
        assert len(doc.spans["findings"]) == 1
        assert doc.spans["findings"][0]._.date == expected_date(
            "20/05/2016", "2016-05-20"
        )

    def test_only_configured_spans_are_considered(self, nlp_entity):
        # Arrange
        doc = nlp_entity("20/05/2016 ENTITY ENTITY")
        first, second = doc.spans["ents"]
        doc.spans["findings"] = [first]
        linker = DateLinker(spans_key="findings")
        stale = expected_date("01/01/2000", "2000-01-01")
        second._.date = stale.copy()

        # Act
        linker(doc)

        # Assert
        assert first._.date == expected_date("20/05/2016", "2016-05-20")
        assert second._.date == stale

    def test_preserves_spans_qualifiers_and_text(self, nlp_entity):
        # Arrange
        text = "20/05/2016\tENTITY en ENTITY"
        doc = nlp_entity(text)
        original_spans = list(doc.spans["ents"])
        doc.spans["other"] = [doc[:]]
        other_spans = list(doc.spans["other"])
        doc.ents = [original_spans[0]]
        original_ents = tuple(doc.ents)
        qualifiers = {Qualifier("Negation", "Negated", is_default=False)}
        for span in original_spans:
            set_qualifiers(span, qualifiers.copy())

        # Act
        result = DateLinker()(doc)

        # Assert
        assert result is doc
        assert doc.text == text
        assert list(doc.spans["ents"]) == original_spans
        assert list(doc.spans["other"]) == other_spans
        assert tuple(doc.ents) == original_ents
        for span in original_spans:
            assert get_qualifiers(span) == qualifiers
            assert span._.date == expected_date("20/05/2016", "2016-05-20")

    @pytest.mark.parametrize(
        "text",
        [
            "ENTITY",
            "32/02/2016 ENTITY",
            "20/05/2016 ENTITY 21/05/2016",
            "20/05/2016\nENTITY",
            "20/05/2016" + " " * 201 + "ENTITY",
        ],
    )
    def test_clears_stale_date_when_no_link_is_possible(self, nlp_entity, text):
        # Arrange
        doc = nlp_entity(text)
        linker = DateLinker()
        span = doc.spans["ents"][0]
        span._.date = expected_date("01/01/2000", "2000-01-01")

        # Act
        result = linker(doc)

        # Assert
        assert result is doc
        assert span._.date is None

    @pytest.mark.parametrize(
        ("date_text", "value", "precision"),
        [
            ("20/05/2016", "2016-05-20", "day"),
            ("31/02/2016", None, "ambiguous"),
            ("1900-02-29", None, "ambiguous"),
            ("30/02", None, "ambiguous"),
            ("04-31", None, "ambiguous"),
        ],
    )
    def test_reprocessing_overwrites_stale_date_and_is_idempotent(
        self, nlp_entity, date_text, value, precision
    ):
        # Arrange
        doc = nlp_entity(f"{date_text} ENTITY")
        linker = DateLinker()
        span = doc.spans["ents"][0]
        span._.date = expected_date("01/01/2000", "2000-01-01")
        expected = expected_date(date_text, value, precision=precision)

        # Act & Assert
        assert linker(doc) is doc
        assert span._.date == expected
        assert linker(doc) is doc
        assert span._.date == expected

    def test_reprocessing_with_shorter_distance_clears_previous_link(self, nlp_entity):
        # Arrange
        doc = nlp_entity("20/05/2016 ENTITY")
        DateLinker()(doc)
        span = doc.spans["ents"][0]
        assert span._.date == expected_date("20/05/2016", "2016-05-20")

        # Act
        DateLinker(max_distance=0)(doc)

        # Assert
        assert span._.date is None

    @pytest.mark.parametrize("spans_key", ["ents", "findings"])
    def test_missing_span_group_is_noop(self, nlp, spans_key):
        # Arrange
        doc = nlp("20/05/2016 ENTITY")
        doc.spans["other"] = [doc[-1:]]
        linker = DateLinker(spans_key=spans_key)
        stale = expected_date("01/01/2000", "2000-01-01")
        doc.spans["other"][0]._.date = stale.copy()
        original = doc.to_bytes()

        # Act
        result = linker(doc)

        # Assert
        assert result is doc
        assert spans_key not in doc.spans
        assert doc.to_bytes() == original
        assert doc.spans["other"][0]._.date == stale

    @pytest.mark.parametrize("text", ["", "20/05/2016"])
    def test_empty_span_group_is_noop(self, nlp, text):
        # Arrange
        doc = nlp(text)
        doc.spans["ents"] = []

        # Act
        result = DateLinker()(doc)

        # Assert
        assert result is doc
        assert doc.text == text
        assert list(doc.spans["ents"]) == []


class TestDateLinkerSerialization:
    def test_doc_bytes_roundtrip(self, nlp_entity):
        # Arrange
        nlp_entity.add_pipe("clinlp_date_linker")
        text = "20/05/2016 ENTITY\nENTITY"
        doc = nlp_entity(text)

        # Act
        restored = Doc(nlp_entity.vocab).from_bytes(doc.to_bytes())

        # Assert
        assert restored.text == text
        assert [(span.text, span.label_) for span in restored.spans["ents"]] == [
            ("ENTITY", "named_entity"),
            ("ENTITY", "named_entity"),
        ]
        assert [
            (span.start_char, span.end_char) for span in restored.spans["ents"]
        ] == [(span.start_char, span.end_char) for span in doc.spans["ents"]]
        assert type(restored.spans["ents"][0]._.date) is dict
        assert [span._.date for span in restored.spans["ents"]] == [
            expected_date("20/05/2016", "2016-05-20"),
            None,
        ]

    @pytest.mark.parametrize("n_process", [1, 2])
    def test_nlp_pipe(self, nlp_entity, n_process):
        # Arrange
        nlp_entity.add_pipe("clinlp_date_linker")
        texts = [
            "20/05/2016 ENTITY",
            "ENTITY",
            "ENTITY 2017-06-21",
            "20/05/2016 ENTITY 21/05/2016",
            "20/05/2016\r\nENTITY",
        ]

        # Act
        docs = list(nlp_entity.pipe(texts, batch_size=2, n_process=n_process))

        # Assert
        assert [doc.text for doc in docs] == texts
        assert [len(doc.spans["ents"]) for doc in docs] == [1] * len(texts)
        assert [doc.spans["ents"][0]._.date for doc in docs] == [
            expected_date("20/05/2016", "2016-05-20"),
            None,
            expected_date("2017-06-21", "2017-06-21", 7),
            None,
            None,
        ]
        assert type(docs[0].spans["ents"][0]._.date) is dict

    def test_pipeline_disk_roundtrip(self, nlp, tmp_path):
        # Arrange
        ruler = nlp.add_pipe("span_ruler", config={"spans_key": "findings"})
        ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])
        nlp.add_pipe(
            "clinlp_date_linker",
            config={"spans_key": "findings", "max_distance": 3},
        )
        path = tmp_path / "date_pipeline"
        texts = ["20/05/2016   ENTITY", "20/05/2016    ENTITY", "ENTITY"]
        expected = [expected_date("20/05/2016", "2016-05-20"), None, None]
        original_docs = list(nlp.pipe(texts))
        assert [doc.spans["findings"][0]._.date for doc in original_docs] == expected

        # Act
        nlp.to_disk(path)
        restored = spacy.load(path)
        docs = list(restored.pipe(texts))

        # Assert
        assert restored.pipe_names == nlp.pipe_names
        assert isinstance(restored.get_pipe("clinlp_date_linker"), DateLinker)
        config = restored.config["components"]["clinlp_date_linker"]
        assert config["spans_key"] == "findings"
        assert config["max_distance"] == 3
        assert [doc.text for doc in docs] == texts
        assert [len(doc.spans["findings"]) for doc in docs] == [1, 1, 1]
        assert [doc.spans["findings"][0]._.date for doc in docs] == expected
        assert all("ents" not in doc.spans for doc in docs)
