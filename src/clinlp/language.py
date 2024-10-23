"""
Language class for the ``clinlp`` package.

It is based on the ``spaCy`` language class, but with custom settings for
Dutch clinical text.
"""

import importlib.metadata
import warnings
from collections.abc import Callable

import spacy.lang.char_classes
import spacy.lang.nl.tokenizer_exceptions
import spacy.lang.punctuation
import spacy.lang.tokenizer_exceptions
from spacy.language import BaseDefaults, Language
from spacy.symbols import ORTH
from spacy.util import update_exc

from clinlp.exceptions import VersionMismatchWarning

CLINLP_ABBREVIATIONS = [
    "dhr.",
    "dr.",
    "dh.",
    "drs.",
    "ds.",
    "gg.",
    "ggn.",
    "grg.",
    "ing.",
    "ir.",
    "medepat.",
    "mej.",
    "mevr.",
    "mr.",
    "mw.",
    "prof.",
    "pt.",
    "1.",
    "2.",
    "3.",
    "4.",
    "5.",
    "6.",
    "7.",
    "8.",
    "9.",
    "10.",
    "11.",
    "12.",
    "I.",
    "II.",
    "III.",
    "IV.",
    "V.",
    "VI.",
    "VII.",
    "VIII.",
    "IX.",
    "X.",
    "t.b.v.",
    "a.n.",
    "internat.eenh.",
    "o.a.",
    "oa.",
    "tel.",
    "i.o.m.",
    "z.n.",
    "o.l.v.",
    "z.n.",
    "e.c.i.",
    "s.c.",
    "etc.",
    "g.b.",
    "t.p.v.",
    "vlgs.",
    "o.b.v.",
    "t.h.v.",
    "i.o.m.",
    "i.v.",
    "internat.eenh.",
    "m.i.",
    "v.v.",
    "i.p.",
    "i.v.m.",
    "i.c.m.",
    "m.b.v.",
    "wv.",
    "pol.",
    "tr.",
    "vv.",
    "mi.",
]

CLINLP_REMOVE_ABBREVIATIONS = [
    "ts.",
]

CLINLP_UNITS = [
    "cc",
    "CC",
    "l",
    "L",
    "ml",
    "mL",
    "µl",
    "µL",
    "dag",
    "h",
    "hh",
    "hr",
    "j",
    "jr",
    "s",
    "sec",
    "W",
    "w",
    "week",
    "wkn",
    "wk",
    "min",
    "mnd",
    "ms",
    "pm",
    "g",
    "gr",
    "mg",
    "mG",
    "µg",
    "ug",
    "mcg",
    "do",
    "dosis",
    "stuk",
    "e",
    "x",
    "Gy",
    "eh",
    "ie",
    "IE",
    "Ie",
    "IE",
    "mol",
    "mmol",
    "umol",
    "µmol",
    "pmol",
    "u",
    "kcal",
    "kCal",
    "m2",
    "µm",
    "mmhg",
    "mmHg",
    "km",
    "km²",
    "km³",
    "m",
    "m²",
    "m³",
    "dm",
    "dm²",
    "dm³",
    "cm",
    "cm²",
    "cm³",
    "mm",
    "mm²",
    "mm³",
    "ha",
    "µm",
    "nm",
    "yd",
    "in",
    "ft",
    "kg",
    "g",
    "mg",
    "µg",
    "t",
    "lb",
    "oz",
    "m/s",
    "km/h",
    "kmh",
    "mph",
    "hPa",
    "Pa",
    "mbar",
    "mb",
    "MB",
    "kb",
    "KB",
    "gb",
    "GB",
    "tb",
    "TB",
    "T",
    "G",
    "M",
    "K",
    "%",
    "км",
    "км²",
    "км³",
    "м",
    "м²",
    "м³",
    "см",
    "см²",
    "см³",
    "мм",
    "мм²",
    "мм³",
    "нм",
]

CLINLP_TOKENIZER_EXCEPTIONS = {
    "\n": [{ORTH: "\n"}],
    "\r": [{ORTH: "\r"}],
    "\t": [{ORTH: "\t"}],
    "->": [{ORTH: "->"}],
    "-->": [{ORTH: "-->"}],
    "<->": [{ORTH: "<->"}],
    "=>": [{ORTH: "=>"}],
    "==>": [{ORTH: "==>"}],
    "<=>": [{ORTH: "<=>"}],
    "+/-": [{ORTH: "+/-"}],
    "t/m": [{ORTH: "t/m"}],
    "<=": [{ORTH: "<="}],
    ">=": [{ORTH: ">="}],
    "(?)": [{ORTH: "(?)"}],
    "xdd": [{ORTH: "x"}, {ORTH: "dd"}],
}

CLINLP_ABBREV_TRANSFORMS = [
    lambda x: x,
    lambda x: x.upper(),
    lambda x: x.capitalize(),
]

ALPHA_LOWER = "a-z"
ALPHA_UPPER = "A-Z"
ALPHA = "a-zA-Z"


def _get_abbreviations() -> list[str]:
    """
    Get the list of abbreviations for ``clinlp``.

    Returns
    -------
    ``list[str]``
        The list of abbreviations.
    """
    base = set(spacy.lang.nl.tokenizer_exceptions.abbrevs.copy())

    for abbrev in CLINLP_REMOVE_ABBREVIATIONS:
        base.remove(abbrev)

    return list(base) + CLINLP_ABBREVIATIONS


def _get_tokenizer_exceptions(
    abbreviations: list[str],
    *,
    abbrev_transforms: list[Callable[[str], str]] | None = None,
    keep_emoticons: bool = False,
) -> dict[str, list[dict]]:
    """
    Get the tokenizer exceptions for ``clinlp``.

    Tokenizer exceptions are sequences of characters that should not be split up.

    Parameters
    ----------
    abbreviations
        All abbreviations, which will be added to the tokenizer exceptions.
    abbrev_transforms
        Any number of callables that transform abbreviations. The transformed versions
        will also be added to the tokenizer exceptions.
    keep_emoticons
        Wether to keep emoticons as tokenizer exceptions. Emoticons (e.g. ``:-)`` are
        normally included in spaCy's base exceptions.

    Returns
    -------
    ``dict[str, list[dict]]``
        The tokenizer exceptions.
    """
    tokenizer_exceptions = spacy.lang.tokenizer_exceptions.BASE_EXCEPTIONS.copy()
    abbrev_transforms = abbrev_transforms or []

    if not keep_emoticons:
        for emoticon in spacy.lang.tokenizer_exceptions.emoticons:
            del tokenizer_exceptions[emoticon]

    for abbrev_transform in abbrev_transforms:
        abbr_update = {
            abbrev_transform(a): [{ORTH: abbrev_transform(a)}] for a in abbreviations
        }
        tokenizer_exceptions = update_exc(tokenizer_exceptions, abbr_update)

    return update_exc(tokenizer_exceptions, CLINLP_TOKENIZER_EXCEPTIONS)


def _get_list(
    base: list[str], add: list[str] | None = None, remove: list[str] | None = None
) -> list[str]:
    """
    Create a list of strings, by copying a base list and adding and/or removing items.

    Parameters
    ----------
    base
        The base list, with the initial items.
    add
        Any items that will be added.
    remove
        Any items that will be removed.

    Returns
    -------
    ``list[str]``
        A new list (copy of the base list) with the added and removed items.
    """
    _lst = base.copy()

    if add is not None:
        for item in add:
            _lst.append(item)

    if remove is not None:
        for item in remove:
            _lst.remove(item)

    return _lst


def _get_ellipses() -> list[str]:
    """
    Get the list of ellipses for ``clinlp``.

    Returns
    -------
    ``list[str]``
        The list of ellipses.
    """
    return spacy.lang.punctuation.LIST_ELLIPSES.copy()


def _get_currencies() -> list[str]:
    """
    Get the list of currencies (e.g. ``$``, ``€``) for ``clinlp``.

    Returns
    -------
    ``list[str]``
        The list of currencies.
    """
    return spacy.lang.punctuation.LIST_CURRENCY.copy()


def _get_units() -> list[str]:
    """
    Get the list of units for ``clinlp``.

    Returns
    -------
    ``list[str]``
        The list of units.
    """
    return CLINLP_UNITS.copy()


def _get_tokenizer_prefix_rules() -> list[str]:
    """
    Get the list of prefix rules for the ``clinlp`` tokenizer.

    Prefix rules are regular expressions that match the start of a token. If the
    regular expression matches, the prefix is split into a separate token.

    Returns
    -------
    ``list[str]``
        The list of prefix rules.
    """
    return [
        r"\[(?![A-Z]{3,}-)",
        r"\S+(?=\[[A-Z]{3,}-)",
        r"x(?=[0-9]+)",
        r"`(?=[0-9])",
        r"([0-9]{,5}(\.|,))?[0-9]{,4}" + f"(?=({'|'.join(_get_units())}))",
    ]


def _get_tokenizer_prefixes() -> list[str]:
    """
    Get the list of prefixes for the ``clinlp`` tokenizer.

    Prefixes are literal strings/chars that are split of the beginning of a token.

    Returns
    -------
    ``list[str]``
        The list of prefixes.
    """
    punct = _get_list(
        base=spacy.lang.punctuation.LIST_PUNCT,
        add=[
            ",,",
            "§",
            "%",
            "=",
            "—",
            "–",  # noqa RUF001
            r"\+(?![0-9])",
            "/",
            "-",
            r"\+",
            "~",
            ",",
            r"\s",
        ],
        remove=[r"\["],
    )

    quotes = _get_list(base=spacy.lang.punctuation.LIST_QUOTES, remove=["`"])

    return (
        punct
        + _get_ellipses()
        + quotes
        + _get_currencies()
        + _get_tokenizer_prefix_rules()
    )


def _get_tokenizer_infix_rules(quotes: list[str]) -> list[str]:
    """
    Get the list of infix rules for the ``clinlp`` tokenizer.

    Infix rules are regular expressions that match the middle of a token. If the
    regular expression matches, the infix is split into a separate token.

    Parameters
    ----------
    quotes
        The list of quotes that are used in the infix rules text.

    Returns
    -------
    ``list[str]``
        The list of infix rules.
    """
    return [
        rf"(?<=[{ALPHA_LOWER}])\.(?=[{ALPHA_UPPER}])",
        rf"(?<=[{ALPHA}])[,!?](?=[{ALPHA}])",
        rf'(?<=[{ALPHA}"])[:<>=](?=[{ALPHA}])',
        rf"(?<=[{ALPHA}]),(?=[{ALPHA}])",
        r"(?<=[{a}])([{q}\)\]\(\[])(?=[{a}])".format(a=ALPHA, q="".join(quotes)),
        rf"(?<=[{ALPHA}])--(?=[{ALPHA}])",
        r"-(?![0-9]{1,2}\])",
        r"(?<=[0-9])x(?=[0-9])",
        r"(?<=10)E(?=\d)",
        r"(?<=[0-9])(x?d?d)(?=[0-9])",
    ]


def _get_tokenizer_infixes() -> list[str]:
    """
    Get the list of infixes for the ``clinlp`` tokenizer.

    Infixes are literal strings/chars that are split in the middle of a token.

    Returns
    -------
    ``list[str]``
        The list of infixes.
    """
    punct = _get_list(
        base=[],
        add=["/", r"\+", "=", ":", "&", ";", r"\*", "<", r"\(", r"\)", r"\s", r"\^"],
    )

    quotes = _get_list(base=spacy.lang.punctuation.LIST_QUOTES, remove=["`", r"\'"])

    return punct + _get_ellipses() + _get_tokenizer_infix_rules(quotes)


def _get_tokenizer_suffix_rules(
    currencies: list[str], units: list[str], punct: list[str], quotes: list[str]
) -> list[str]:
    """
    Get the list of suffix rules for the ``clinlp`` tokenizer.

    Suffix rules are regular expressions that match the end of a token. If the
    regular expression matches, the suffix is split into a separate token.

    Parameters
    ----------
    currencies
        A list of currencies.
    units
        A list of units.
    punct
        A list of punctuation.
    quotes
        A list of quotes.

    Returns
    -------
    ``list[str]``
        The list of suffix rules.
    """
    return [
        r"(?<=[0-9])\+",
        r"(?<=°[FfCcKk])\.",
        r"(?<=[0-9])(?:{c})".format(c="|".join(currencies)),
        r"(?<=[0-9{al}{e}{p}(?:{q})])\.".format(
            al=ALPHA_LOWER, e=r"%²\-\+", q="".join(quotes), p="|".join(punct)
        ),
        rf"(?<=[{ALPHA_UPPER}][{ALPHA_UPPER}])\.",
        r"(?<=[0-9]\])\S+",
        r"(?<!([A-Z]-\d|-\d\d))\]",  # tricky one
        r"(?<=[0-9])x",
        r"(?<=[0-9])" + f"({'|'.join(units)})",
        r"\s",
        r"(?<=[0-9])d?d",
        r"(?<=[0-9])(e|de|ste)",
    ]


def _get_tokenizer_suffixes() -> list[str]:
    """
    Get a list of suffixes for the ``clinlp`` tokenizer.

    Suffixes are literal strings/chars that are split at the end of a token.

    Returns
    -------
    ``list[str]``
        The list of suffixes.
    """
    punct = _get_list(
        base=spacy.lang.punctuation.LIST_PUNCT,
        add=["/", "-", "=", "%", r"\+", "~", "''", "—", "–"],  # noqa RUF001
        remove=[r"\]"],
    )

    quotes = _get_list(base=spacy.lang.punctuation.LIST_QUOTES)

    return (
        punct
        + quotes
        + _get_ellipses()
        + _get_tokenizer_suffix_rules(
            currencies=_get_currencies(), units=_get_units(), punct=punct, quotes=quotes
        )
    )


class ClinlpDefaults(BaseDefaults):
    """Default settings for the ``clinlp`` language class."""

    tokenizer_exceptions = _get_tokenizer_exceptions(
        abbreviations=_get_abbreviations(), abbrev_transforms=CLINLP_ABBREV_TRANSFORMS
    )

    prefixes = _get_tokenizer_prefixes()
    infixes = _get_tokenizer_infixes()
    suffixes = _get_tokenizer_suffixes()

    lex_attr_getters = {}  # noqa: RUF012
    syntax_iterators = {}  # noqa: RUF012
    stop_words = set()  # noqa: RUF012
    url_match = None
    token_match = None

    writing_system = {  # noqa: RUF012
        "direction": "ltr",
        "has_case": True,
        "has_letters": True,
    }


@spacy.registry.languages("clinlp")
class Clinlp(Language):
    """
    ``clinlp`` language class.

    Contains custom settings for Dutch clinical text.
    """

    lang = "clinlp"
    Defaults = ClinlpDefaults

    def __init__(self, *args, **kwargs) -> None:
        """Create a ``clinlp`` language object."""
        meta = dict(kwargs.pop("meta", {}))
        clinlp_version = importlib.metadata.version(__package__ or __name__)

        if "clinlp_version" in meta:
            if meta["clinlp_version"] != clinlp_version:
                warnings.warn(
                    f"This spaCy model was built with clinlp version "
                    f"{meta['clinlp_version']}, but you currently have version "
                    f"{clinlp_version} installed, potentially leading to unexpected "
                    f"results.",
                    VersionMismatchWarning,
                    stacklevel=2,
                )
        else:
            meta["clinlp_version"] = clinlp_version

        super().__init__(*args, meta=meta, **kwargs)
