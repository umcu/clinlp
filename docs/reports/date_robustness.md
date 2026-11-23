# Numeric date-linker robustness check

> Historical assessment: the results below describe the earlier strict-calendar
> implementation. Extraction was subsequently made permissive at the user's request:
> day 1–31 / month 1–12 expressions such as `31/04/2020` and `31/04` are now retained
> with `precision="ambiguous"`, while out-of-range components remain rejected.
> Recognition has since also expanded to Dutch month names, unpadded/year-first
> dates, US order, two-digit years, bare years, whitespace, Unicode separators and
> attached timestamps. Competing full day/month orders are now marked ambiguous.
> See `source/components.md` for the current behavior. The regression tests have
> been updated; unsupported-format examples and counts below describe the original
> assessment, not the current implementation.

Run date: 2026-11-05. This is a synthetic regression and edge-case assessment,
not a clinical validation or an estimate of precision/recall on patient reports.
No production extraction rules were changed during this assessment.

## Completed validation

- **315 clinlp tests passed**, including 82 new grouped robustness tests.
- The new matrix exercises **9,532 inputs** (including repeated padding variants).
- **32 RBSE date-aggregation tests passed**, covering single/bulk parity, qualifier
  filtering, deduplication, partial metadata and undated counts.
- Ruff lint and formatting checks passed for both clinlp date-test files.
- One existing multiprocessing case (`test_nlp_pipe[2]`) was deselected. Serial
  `nlp.pipe`, Doc byte serialization and pipeline save/load were tested.

Tests used the existing Python 3.13 environment and editable local clinlp. Optional
transformer discovery was disabled only in the test process to avoid unrelated
model startup; transformer classification and multiprocessing were not validated.
The new matrix inserts known concept spans to isolate date linking; RBSE tests use
real concept matching. The exploratory examples below used an exact
`PD peritonitis` concept matcher without qualifier filtering.

An existing serialization test discarded its temporary Doc before accessing its
SpanGroup, triggering spaCy E865. The test now retains its Docs. This was a test
lifetime issue, not a date-parser change. Two generated negative-test expectations
were also corrected: `1/2/2000` and `1-02-2000` are valid full dates.

## Matrix coverage

| Category | Input checks |
| --- | ---: |
| Full dates: 1900/2000/2024, months 0–13, days 0–32, four formats | 5,544 |
| Additional year extremes, leap-century rules and component widths | 360 |
| Short pairs: both components 0–32, `/` and `-`, plus padding variants | 2,226 |
| Month/year and year/month, both separators and month widths | 672 |
| Punctuation, before/after positions and character offsets | 100 |
| CR, LF, CRLF isolation and multiline spans | 210 |
| Inclusive distance boundaries and zero-gap spans | 200 |
| Date/concept overlap and multiple date expressions | 40 |
| Embedded identifiers and malformed numeric sequences | 164 |
| Currently unsupported formats | 16 |
| **Total** | **9,532** |

Expected calendar validity comes from Python's `datetime`/`calendar`, independently
of the production regex. Invalid full dates must not fall back to a partial match.
Assertions also check original date text, precision, offsets and unchanged spans.

## Supported behavior verified

| Example | Linked value | Precision |
| --- | --- | --- |
| `20/09/2020`, `20-09-2020`, `20.09.2020` | `2020-09-20` | day |
| `2020-09-20` | `2020-09-20` | day |
| `2020-9:`, `2020/09`, `9/2020`, `09-2020` | `2020-09` | month |
| `09/10`, `20/09`, `09/20`, `29/02` | null | ambiguous |
| `29/02/1900`, `31/04/2020`, `31/04`, `13/14` | No link | — |

Ambiguous expressions retain their raw text. RBSE includes that raw text in
`dates`, while preserving `value=None` and `precision="ambiguous"` in
`partial_dates`. Known month dates appear in `dates` as normalized `YYYY-MM`.
No missing year or day is invented.

`xxxx 2020-9: PD peritonitis xxxxx` links `2020-9` at character offsets `[5, 11)`
to `2020-09` with month precision. Historical qualifier filtering is separate and
can still exclude this mention from RBSE aggregation.

## Exploratory findings and limitations

### False-positive candidates

All of these produced ambiguous date metadata, despite the numerical expressions
having plausible non-date meanings:

| Synthetic text | Captured expression |
| --- | --- |
| `Pijnscore 9/10. PD peritonitis` | `9/10` |
| `Dosis 1/2 tablet. PD peritonitis` | `1/2` |
| `Stadium 2-3. PD peritonitis` | `2-3` |

The component links across sentence punctuation when expressions are on the same
physical line. `precision="ambiguous"` signals unresolved information, not proof
that the expression is a date.

In `20/09/2020 PD peritonitis 1/2 tablet`, **no date was linked**: the dose resembles
a second date expression, triggering the conservative multiple-expression rule.
Thus short-pair recognition can both add false-positive candidates and suppress
otherwise valid full-date associations.

### Unsupported formats observed

Each of the following, placed before `PD peritonitis`, produced no date link:

- `20 mei 2020` (written month).
- `2020-9-1` (unpadded year-first full date).
- `2020/09/01` (slash-separated year-first full date).
- `09/20/2020` (US month-first full date).
- `20/09/20` (two-digit year).
- `2020` (bare year).
- `20 / 09 / 2020` (spaces around separators).
- `2020–9` (Unicode en dash).
- `2020-09-01T12:00` (attached timestamp).

Full day-first dates retain their configured interpretation: `05/06/2020`
normalizes to `2020-06-05`, not an ambiguous value. Full US/Dutch order is not
inferred from surrounding context.

### Boundary limitations observed

- `-2020-9: PD peritonitis` is not linked; `- 2020-9: PD peritonitis` is linked.
  The numeric-sequence guard also blocks a directly attached hyphen bullet.
- `2020-9\u2028PD peritonitis`, with an actual Unicode line separator, is linked.
  Only CR and LF delimit physical lines for the linker; Unicode separators are not
  currently treated as boundaries. The CR/LF tests passing does not cover this.

## Interpretation and next priorities

The implementation is consistent over its supported numeric grammar and calendar
rules. The main remaining risks are contextual, not calendar arithmetic:

1. Distinguish scores, doses and ranges from short date candidates, especially when
   they compete with a full date on the same line.
2. Consider adding whitespace-tolerant dates and unpadded year-first full dates.
3. Decide explicit policies for bare years, written months, two-digit years and
   month-first full dates rather than silently assuming an interpretation.
4. Normalize or explicitly support Unicode line separators/dashes and list bullets.

These are proposed follow-up changes, not implemented by this test-only task.
Repeated ambiguous date strings and overlapping concept spans must not be treated
as unique clinical events without downstream reconciliation.

## Re-running

With clinlp and its test dependencies installed, from the clinlp repository:

```sh
python -m pytest tests/unit/ie/test_date.py tests/unit/ie/test_date_robustness.py --no-cov -k "not (test_nlp_pipe and 2)"
```

From the RBSE repository, with the updated editable clinlp installed:

```sh
python -m unittest discover -s tests -p test_extractor_dates.py -v
```

These commands use normal package discovery. The reported environment additionally
used a process-local transformer-discovery override; it did not change package
files or persistent environment settings. Remove the pytest selection expression
to include multiprocessing when the environment supports running that case.
