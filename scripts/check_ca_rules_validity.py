"""
Checks a file with rules for the context algorithm for some common errors
and inconsistencies, and shows them in the terminal (if any).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

RULE_FILE = "src/clinlp/resources/context_rules.json"


def get_patterns_from_direction(rules: list[dict], direction: str) -> list[dict]:
    for rule in rules:
        if rule["direction"] == direction:
            return rule["patterns"]

    return []


def any_in_pseudo(pseudo_term: str, patterns: set) -> bool:
    return any(pattern in pseudo_term for pattern in patterns)


if __name__ == "__main__":
    with Path(RULE_FILE).open() as f:
        data = json.load(f)

    for rule in data["rules"]:
        rule["patterns"] = [
            pattern for pattern in rule["patterns"] if isinstance(pattern, str)
        ]

    print("Checking for duplicates...")

    for rule in data["rules"]:
        c = Counter()
        c.update(rule["patterns"])
        c = {k: n for k, n in c.items() if n != 1}

        if len(c) > 0:
            print("Dupicates:", rule["qualifier"], rule["direction"], c)

    print("Checking sort order...")
    for rule in data["rules"]:
        if sorted(rule["patterns"]) != rule["patterns"]:
            print("Needs sorting: ", rule["qualifier"], rule["direction"])

    print("Checking trim errors...")
    for rule in data["rules"]:
        for pattern in rule["patterns"]:
            if pattern.strip() != pattern:
                print("Needs trimming: ", rule["qualifier"], rule["direction"], pattern)

    print("Checking overlapping patterns...")

    grouped = defaultdict(list)
    for rule in data["rules"]:
        grouped[rule["qualifier"]].append(rule)

    for name, rules in grouped.items():
        preceding = set(get_patterns_from_direction(rules, "preceding"))
        following = set(get_patterns_from_direction(rules, "following"))
        bidirectional = set(get_patterns_from_direction(rules, "bidirectional"))
        pseudo = set(get_patterns_from_direction(rules, "pseudo"))
        termination = set(get_patterns_from_direction(rules, "termination"))

        def print_if_nonempty(name: str, items: set) -> None:
            if len(items) > 0:
                print("Overlap:", name, items)

        print_if_nonempty(name, preceding.intersection(following))
        print_if_nonempty(name, preceding.intersection(bidirectional))
        print_if_nonempty(name, following.intersection(bidirectional))
        print_if_nonempty(
            name, preceding.union(following).union(bidirectional).intersection(pseudo)
        )
        print_if_nonempty(
            name,
            preceding.union(following).union(bidirectional).intersection(termination),
        )

    print("Checking spurious pseudo patterns...")

    for name, rules in grouped.items():
        preceding = set(get_patterns_from_direction(rules, "preceding"))
        following = set(get_patterns_from_direction(rules, "following"))
        bidirectional = set(get_patterns_from_direction(rules, "bidirectional"))
        pseudo = set(get_patterns_from_direction(rules, "pseudo"))

        all_others = preceding.union(following).union(bidirectional)

        for p in pseudo:
            if not any_in_pseudo(p, all_others):
                print("Spurious pseudo pattern:", name, p)
