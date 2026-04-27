"""Run a sequence of rules over a corpus and collect statistics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from collections import Counter

from .sentence import Sentence
from .rules.base import Rule, Change
from .rules.advcl_pred import AdvclPredRule
from .rules.obl_agent import OblAgentRule
from .rules.obl_arg import OblArgRule


DEFAULT_PIPELINE: List[Rule] = [
    AdvclPredRule(),
    OblAgentRule(),
    OblArgRule(),
]


@dataclass
class RunReport:
    changes: List[Change] = field(default_factory=list)
    rule_counts: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        lines = [f"Total changes: {len(self.changes)}", ""]
        lines.append("Changes by rule:")
        for name, n in self.rule_counts.most_common():
            lines.append(f"  {name}: {n}")
        return "\n".join(lines)


def run_pipeline(sentences: List[Sentence],
                 rules: List[Rule] = None) -> RunReport:
    if rules is None:
        rules = DEFAULT_PIPELINE
    report = RunReport()
    for sent in sentences:
        for rule in rules:
            for change in rule.apply(sent):
                report.changes.append(change)
                report.rule_counts[change.rule_name] += 1
    return report