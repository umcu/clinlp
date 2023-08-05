import importlib.metadata
import warnings
from typing import Callable, Optional

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


def _get_abbreviations():
    base = set(spacy.lang.nl.tokenizer_exceptions.abbrevs.copy())

    for abbrev in CLINLP_REMOVE_ABBREVIATIONS:
        base.remove(abbrev)

    return list(base) + CLINLP_ABBREVIATIONS


def _get_tokenizer_exceptions(
    abbreviations: list[str],
    abbrev_transforms: list[Callable[[str], str]] = None,
    keep_emoticons: bool = False,
):
    tokenizer_exceptions = spacy.lang.tokenizer_exceptions.BASE_EXCEPTIONS.copy()

    if not keep_emoticons:
        for emoticon in spacy.lang.tokenizer_exceptions.emoticons:
            del tokenizer_exceptions[emoticon]

    for abbrev_transform in abbrev_transforms:
        abbr_update = {abbrev_transform(a): [{ORTH: abbrev_transform(a)}] for a in abbreviations}
        tokenizer_exceptions = update_exc(tokenizer_exceptions, abbr_update)

    return update_exc(tokenizer_exceptions, CLINLP_TOKENIZER_EXCEPTIONS)


def _get_list(base: list[str], add: Optional[list[str]] = None, remove: Optional[list[str]] = None):
    """
    Create a list, based on a base list, with some additions and removals.
    """
    _lst = base.copy()

    if add is not None:
        for item in add:
            _lst.append(item)

    if remove is not None:
        for item in remove:
            _lst.remove(item)

    return _lst


def _get_ellipses():
    return spacy.lang.punctuation.LIST_ELLIPSES.copy()


def _get_currencies():
    return spacy.lang.punctuation.LIST_CURRENCY.copy()


def _get_units():
    return CLINLP_UNITS.copy()


def _get_tokenizer_prefix_rules():
    return [
        r"\[(?![A-Z]{3,}-)",
        r"\S+(?=\[[A-Z]{3,}-)",
        r"x(?=[0-9]+)",
        r"`(?=[0-9])",
        r"([0-9]{,5}(\.|,))?[0-9]{,4}" + f"(?=({'|'.join(_get_units())}))",
    ]


def _get_tokenizer_prefixes():
    punct = _get_list(
        base=spacy.lang.punctuation.LIST_PUNCT,
        add=[",,", "§", "%", "=", "—", "–", r"\+(?![0-9])", "/", "-", r"\+", "~", ",", r"\s"],
        remove=[r"\["],
    )

    quotes = _get_list(base=spacy.lang.punctuation.LIST_QUOTES, remove=["`"])

    return punct + _get_ellipses() + quotes + _get_currencies() + _get_tokenizer_prefix_rules()


def _get_tokenizer_infix_rules(quotes: list[str]):
    return [
        r"(?<=[{al}])\.(?=[{au}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER),
        r"(?<=[{a}])[,!?](?=[{a}])".format(a=ALPHA),
        r'(?<=[{a}"])[:<>=](?=[{a}])'.format(a=ALPHA),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}])([{q}\)\]\(\[])(?=[{a}])".format(a=ALPHA, q="".join(quotes)),
        r"(?<=[{a}])--(?=[{a}])".format(a=ALPHA),
        r"-(?![0-9]{1,2}\])",
        r"(?<=[0-9])x(?=[0-9])",
        r"(?<=10)E(?=\d)",
        r"(?<=[0-9])(x?d?d)(?=[0-9])",
    ]


def _get_tokenizer_infixes():
    punct = _get_list(base=[], add=["/", r"\+", "=", ":", "&", ";", r"\*", "<", r"\(", r"\)", r"\s", r"\^"])

    quotes = _get_list(base=spacy.lang.punctuation.LIST_QUOTES, remove=["`", r"\'"])

    infixes = punct + _get_ellipses() + _get_tokenizer_infix_rules(quotes)

    return infixes


def _get_tokenizer_suffix_rules(currencies: list[str], units: list[str], punct: list[str], quotes: list[str]):
    return [
        r"(?<=[0-9])\+",
        r"(?<=°[FfCcKk])\.",
        r"(?<=[0-9])(?:{c})".format(c="|".join(currencies)),
        r"(?<=[0-9{al}{e}{p}(?:{q})])\.".format(al=ALPHA_LOWER, e=r"%²\-\+", q="".join(quotes), p="|".join(punct)),
        r"(?<=[{au}][{au}])\.".format(au=ALPHA_UPPER),
        r"(?<=[0-9]\])\S+",
        r"(?<!([A-Z]-\d|-\d\d))\]",  # tricky one
        r"(?<=[0-9])x",
        r"(?<=[0-9])" + f"({'|'.join(units)})",
        r"\s",
        r"(?<=[0-9])d?d",
        r"(?<=[0-9])(e|de|ste)",
    ]


def _get_tokenizer_suffixes():
    punct = _get_list(
        base=spacy.lang.punctuation.LIST_PUNCT, add=["/", "-", "=", "%", r"\+", "~", "''", "—", "–"], remove=[r"\]"]
    )

    quotes = _get_list(base=spacy.lang.punctuation.LIST_QUOTES)

    return (
        punct
        + quotes
        + _get_ellipses()
        + _get_tokenizer_suffix_rules(currencies=_get_currencies(), units=_get_units(), punct=punct, quotes=quotes)
    )


class ClinlpDefaults(BaseDefaults):
    tokenizer_exceptions = _get_tokenizer_exceptions(
        abbreviations=_get_abbreviations(), abbrev_transforms=CLINLP_ABBREV_TRANSFORMS
    )

    prefixes = _get_tokenizer_prefixes()
    infixes = _get_tokenizer_infixes()
    suffixes = _get_tokenizer_suffixes()

    lex_attr_getters = {}
    syntax_iterators = {}
    stop_words = []
    url_match = None
    token_match = None

    writing_system = {"direction": "ltr", "has_case": True, "has_letters": True}


@spacy.registry.languages("clinlp")
class Clinlp(Language):
    lang = "clinlp"
    Defaults = ClinlpDefaults

    def __init__(self, *args, **kwargs):
        meta = dict(kwargs.pop("meta", {}))
        clinlp_version = importlib.metadata.version(__package__ or __name__)

        if "clinlp_version" in meta:
            if meta["clinlp_version"] != clinlp_version:
                warnings.warn(
                    f"This spaCy model was built with clinlp version {meta['clinlp_version']}, "
                    f"but you currently have version {clinlp_version} installed, "
                    f"potentially leading to unexpected results.",
                    VersionMismatchWarning,
                )
        else:
            meta["clinlp_version"] = clinlp_version

        super().__init__(*args, meta=meta, **kwargs)
