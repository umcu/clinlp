""" TODO remove hardcoded stuff and make configurable """

import re

from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex

CUSTOM_PREFIXES = [
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

CUSTOM_ABBREVIATIONS = [
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

UNITS = [
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


def make_tokenizer(nlp):
    infixes = nlp.Defaults.infixes.copy()
    prefixes = nlp.Defaults.prefixes.copy()
    suffixes = nlp.Defaults.suffixes.copy()
    tokenizer_exceptions = nlp.Defaults.tokenizer_exceptions.copy()

    # Punctuation
    prefixes.remove(r"`")
    prefixes.append(r"/")
    prefixes.append(r"-")
    prefixes.append(r"\+")
    prefixes.append(r"~")
    prefixes.append(r",")

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

    suffixes.append(r"/")
    suffixes.append(r"-")
    suffixes.append(r"=")
    suffixes.append(r"%")
    suffixes.append(r"\+")
    suffixes.append(r"~")

    # Deduce tags
    prefixes.remove(r"\[")
    prefixes.append(r"\[(?![A-Z]{3,}-)")
    prefixes.append(r"\S+(?=\[[A-Z]{3,})")
    infixes.append(r"-(?![0-9]{1,2}\])")
    suffixes.remove(r"\]")
    suffixes.append(r"(?<=[0-9]\])\S+")
    suffixes.append(r"(?<!([A-Z]-\d|-\d\d))\]")  # tricky one

    # x
    prefixes.append(r"x(?=[0-9]+)")
    infixes.append(r"(?<=[0-9])x(?=[0-9])")
    suffixes.append(r"(?<=[0-9])x")

    # E, ^
    infixes.append(r"(?<=10)E(?=\d)")
    infixes.append(r"\^")

    # `22
    prefixes.append(r"`(?=[0-9])")

    # units
    prefixes.append(r"([0-9]{,5}(\.|,))?[0-9]{,4}" + f"(?=({'|'.join(UNITS)}))")
    suffixes.append(r"(?<=[0-9])" + f"({'|'.join(UNITS)})")

    # newlines, missed whitespaces
    prefixes.append(r"\s")
    infixes.append(r"\s")
    suffixes.append(r"\s")

    # doseringen
    infixes.append(r"(?<=[0-9])(x?d?d)(?=[0-9])")
    suffixes.append(r"(?<=[0-9])d?d")

    # rangtelwoord
    suffixes.append(r"(?<=[0-9])(e|de|ste)")

    # Exceptions
    tokenizer_exceptions = {k: v for k, v in tokenizer_exceptions.items() if re.compile(r"[a-z-A-Z]").search(k)}

    tokenizer_exceptions["->"] = [{65: "->"}]
    tokenizer_exceptions["-->"] = [{65: "-->"}]
    tokenizer_exceptions["<->"] = [{65: "<->"}]
    tokenizer_exceptions["=>"] = [{65: "=>"}]
    tokenizer_exceptions["==>"] = [{65: "==>"}]
    tokenizer_exceptions["<=>"] = [{65: "<=>"}]
    tokenizer_exceptions["+/-"] = [{65: "+/-"}]
    tokenizer_exceptions["t/m"] = [{65: "t/m"}]
    tokenizer_exceptions["<="] = [{65: "<="}]
    tokenizer_exceptions[">="] = [{65: ">="}]
    tokenizer_exceptions["(?)"] = [{65: "(?)"}]

    tokenizer_exceptions["\n"] = [{65: "\n"}]
    tokenizer_exceptions["\r"] = [{65: "\r"}]
    tokenizer_exceptions["\t"] = [{65: "\t"}]

    tokenizer_exceptions["xdd"] = [{65: "x"}, {65: "dd"}]

    for abbr in CUSTOM_ABBREVIATIONS:
        tokenizer_exceptions[abbr] = [{65: abbr}]

    for abbr in CUSTOM_PREFIXES:
        tokenizer_exceptions[abbr] = [{65: abbr}]
        tokenizer_exceptions[abbr.title()] = [{65: abbr.title()}]

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)

    tokenizer = Tokenizer(
        nlp.vocab,
        rules=tokenizer_exceptions,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
    )

    return tokenizer
