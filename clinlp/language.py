import spacy.lang.nl.tokenizer_exceptions
import spacy.lang.tokenizer_exceptions
from spacy.lang.nl import Dutch
from spacy.language import BaseDefaults, Language
from spacy.symbols import ORTH
from spacy.util import update_exc

CLINLP_PREFIXES = [
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
]

CLINLP_ABBREVIATIONS = [
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
]

CLINLP_BASE_EXCEPTIONS = {
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


def _get_tokenizer_exceptions():
    tokenizer_exceptions = spacy.lang.tokenizer_exceptions.BASE_EXCEPTIONS.copy()

    for emoticon in spacy.lang.tokenizer_exceptions.emoticons:
        del tokenizer_exceptions[emoticon]

    tokenizer_exceptions = update_exc(tokenizer_exceptions, CLINLP_BASE_EXCEPTIONS)

    abbreviations = spacy.lang.nl.tokenizer_exceptions.abbrevs + CLINLP_ABBREVIATIONS + CLINLP_PREFIXES

    abbr_transforms = [
        lambda x: x,
        lambda x: x.upper(),
        lambda x: x.capitalize(),
    ]

    for abbr_transform in abbr_transforms:
        abbr_update = {abbr_transform(a): [{ORTH: abbr_transform(a)}] for a in abbreviations}
        tokenizer_exceptions = update_exc(tokenizer_exceptions, abbr_update)

    return tokenizer_exceptions


def _get_tokenizer_prefixes():
    prefixes = Dutch.Defaults.prefixes.copy()

    # Punctuation
    prefixes.remove(r"`")
    prefixes.append(r"/")
    prefixes.append(r"-")
    prefixes.append(r"\+")
    prefixes.append(r"~")
    prefixes.append(r",")

    # Deduce tags
    prefixes.remove(r"\[")
    prefixes.append(r"\[(?![A-Z]{3,}-)")
    prefixes.append(r"\S+(?=\[[A-Z]{3,}-)")

    # x
    prefixes.append(r"x(?=[0-9]+)")

    # `22
    prefixes.append(r"`(?=[0-9])")

    # units
    prefixes.append(r"([0-9]{,5}(\.|,))?[0-9]{,4}" + f"(?=({'|'.join(CLINLP_UNITS)}))")

    # newlines, missed whitespaces
    prefixes.append(r"\s")

    return prefixes


def _get_tokenizer_infixes():
    infixes = Dutch.Defaults.infixes.copy()

    infixes = [item.replace(r"`", "") for item in infixes]
    infixes.append(r"/")
    infixes.append(r"\+")
    infixes.append(r"=")
    infixes.append(r":")
    infixes.append(r"&")
    infixes.append(r";")
    infixes.append(r"\*")
    infixes.append(r"<")
    infixes.append(r"\(")
    infixes.append(r"\)")

    infixes.append(r"-(?![0-9]{1,2}\])")

    # x
    infixes.append(r"(?<=[0-9])x(?=[0-9])")

    # E, ^
    infixes.append(r"(?<=10)E(?=\d)")
    infixes.append(r"\^")

    # newlines, missed whitespaces
    infixes.append(r"\s")

    # doseringen
    infixes.append(r"(?<=[0-9])(x?d?d)(?=[0-9])")

    return infixes


def _get_tokenizer_suffixes():
    suffixes = Dutch.Defaults.suffixes.copy()

    suffixes.append(r"/")
    suffixes.append(r"-")
    suffixes.append(r"=")
    suffixes.append(r"%")
    suffixes.append(r"\+")
    suffixes.append(r"~")

    # Deduce tags
    suffixes.remove(r"\]")
    suffixes.append(r"(?<=[0-9]\])\S+")
    suffixes.append(r"(?<!([A-Z]-\d|-\d\d))\]")  # tricky one

    # x
    suffixes.append(r"(?<=[0-9])x")

    # units
    suffixes.append(r"(?<=[0-9])" + f"({'|'.join(CLINLP_UNITS)})")

    # newlines, missed whitespaces
    suffixes.append(r"\s")

    # doseringen
    suffixes.append(r"(?<=[0-9])d?d")

    # rangtelwoord
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
