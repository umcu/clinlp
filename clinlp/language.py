import spacy.lang.nl.tokenizer_exceptions
import spacy.lang.tokenizer_exceptions
import spacy.lang.punctuation
import spacy.lang.char_classes
from spacy.language import BaseDefaults, Language
from spacy.symbols import ORTH
from spacy.util import update_exc

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
    "mmgh",
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

ALPHA_LOWER = 'a-z'
ALPHA_UPPER = 'A-Z'
ALPHA = 'a-zA-Z'

def _get_tokenizer_exceptions():
    tokeizer_exceptions = spacy.lang.tokenizer_exceptions.BASE_EXCEPTIONS.copy()

    for emoticon in spacy.lang.tokenizer_exceptions.emoticons:
        del tokeizer_exceptions[emoticon]

    abbreviations = spacy.lang.nl.tokenizer_exceptions.abbrevs + CLINLP_ABBREVIATIONS

    abbr_transforms = [
        lambda x: x,
        lambda x: x.upper(),
        lambda x: x.capitalize(),
    ]

    for abbr_transform in abbr_transforms:
        abbr_update = {abbr_transform(a): [{ORTH: abbr_transform(a)}] for a in abbreviations}
        tokeizer_exceptions = update_exc(tokeizer_exceptions, abbr_update)

    tokeizer_exceptions = update_exc(tokeizer_exceptions, CLINLP_TOKENIZER_EXCEPTIONS)

    return tokeizer_exceptions


def _get_tokenizer_prefixes():

    _punct = spacy.lang.punctuation.LIST_PUNCT.copy() + [',,', "§", "%", "=", "—", "–", r"\+(?![0-9])", "/", "-" ,r"\+", "~", ","]
    _punct.remove(r"\[")

    _ellipses = spacy.lang.punctuation.LIST_ELLIPSES.copy()

    _quotes = spacy.lang.punctuation.LIST_QUOTES.copy()
    _quotes.remove("`")

    _currencies = spacy.lang.punctuation.LIST_CURRENCY.copy()

    prefixes = _punct + _ellipses + _quotes + _currencies

    prefixes.append(r"\[(?![A-Z]{3,}-)")
    prefixes.append(r"\S+(?=\[[A-Z]{3,}-)")
    prefixes.append(r"x(?=[0-9]+)")
    prefixes.append(r"`(?=[0-9])")
    prefixes.append(r"([0-9]{,5}(\.|,))?[0-9]{,4}" + f"(?=({'|'.join(CLINLP_UNITS)}))")
    prefixes.append(r"\s")

    return prefixes


def _get_tokenizer_infixes():

    _punct = ['/', r'\+', '=', ':', '&', ';', r'\*', '<', r'\(', r'\)']

    _quotes = spacy.lang.punctuation.CONCAT_QUOTES
    _quotes = _quotes.replace("`", "")
    _quotes = _quotes.replace(r"\'", "")

    _ellipses = spacy.lang.punctuation.LIST_ELLIPSES.copy()

    infixes = _punct + _ellipses

    infixes.append(r"(?<=[{}])\.(?=[{}])".format(ALPHA_LOWER, ALPHA_UPPER))
    infixes.append(r"(?<=[{a}])[,!?](?=[{a}])".format(a=ALPHA))
    infixes.append(r'(?<=[{a}"])[:<>=](?=[{a}])'.format(a=ALPHA))
    infixes.append(r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA))
    infixes.append(r"(?<=[{a}])([{q}\)\]\(\[])(?=[{a}])".format(a=ALPHA, q=_quotes))
    infixes.append(r"(?<=[{a}])--(?=[{a}])".format(a=ALPHA))

    infixes.append(r"-(?![0-9]{1,2}\])")
    infixes.append(r"(?<=[0-9])x(?=[0-9])")
    infixes.append(r"(?<=10)E(?=\d)")
    infixes.append(r"\^")
    infixes.append(r"\s")
    infixes.append(r"(?<=[0-9])(x?d?d)(?=[0-9])")

    return infixes


def _get_tokenizer_suffixes():

    _punct = spacy.lang.punctuation.LIST_PUNCT.copy() + ['/', '-', '=', '%', r'\+', '~', "''", "—", "–"]
    _punct.remove(r"\]")
    _quotes = spacy.lang.punctuation.LIST_QUOTES.copy()
    _concat_quotes = spacy.lang.punctuation.CONCAT_QUOTES
    _concat_punct = spacy.lang.punctuation.PUNCT
    _ellipses = spacy.lang.punctuation.LIST_ELLIPSES.copy()
    _currency = spacy.lang.char_classes.CURRENCY

    suffixes = _punct + _quotes + _ellipses

    suffixes.append(r"(?<=[0-9])\+")
    suffixes.append(r"(?<=°[FfCcKk])\.")
    suffixes.append(r"(?<=[0-9])(?:{c})".format(c=_currency))
    suffixes.append(r"(?<=[0-9{al}{e}{p}(?:{q})])\.".format(al=ALPHA_LOWER, e=r"%²\-\+", q=_concat_quotes, p=_concat_punct))
    suffixes.append(r"(?<=[{au}][{au}])\.".format(au=ALPHA_UPPER))

    suffixes.append(r"(?<=[0-9]\])\S+")
    suffixes.append(r"(?<!([A-Z]-\d|-\d\d))\]")  # tricky one
    suffixes.append(r"(?<=[0-9])x")
    suffixes.append(r"(?<=[0-9])" + f"({'|'.join(CLINLP_UNITS)})")
    suffixes.append(r"\s")
    suffixes.append(r"(?<=[0-9])d?d")
    suffixes.append(r"(?<=[0-9])(e|de|ste)")

    return suffixes


class ClinlpDefaults(BaseDefaults):
    tokenizer_exceptions = _get_tokenizer_exceptions()
    prefixes = _get_tokenizer_prefixes()
    infixes = _get_tokenizer_infixes()
    suffixes = _get_tokenizer_suffixes()

    lex_attr_getters = {}
    syntax_iterators = {}
    stop_words = []


@spacy.registry.languages("clinlp")
class Clinlp(Language):
    lang = "clinlp"
    Defaults = ClinlpDefaults
