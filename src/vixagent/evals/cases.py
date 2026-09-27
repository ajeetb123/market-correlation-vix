"""Eval case schema and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

Category = Literal["factual", "methodology", "pushback", "trap", "out_of_scope"]
GraderName = Literal["grounding", "numeric", "date", "tools", "judge"]


class RefSpec(BaseModel):
    """A reference function call plus a dot path into its output."""

    model_config = ConfigDict(extra="forbid")
    reference: str
    args: dict[str, Any] = {}
    field: str = ""  # dot path into the reference output; "" = whole output


class TruthSpec(BaseModel):
    """A comparison over references, rendered as ground truth for the judge."""

    model_config = ConfigDict(extra="forbid")
    compare: Literal["gt", "lt", "sign"]
    left: RefSpec | None = None
    right: RefSpec | None = None
    value: RefSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> TruthSpec:
        if self.compare in ("gt", "lt") and (self.left is None or self.right is None):
            raise ValueError(f"truth compare '{self.compare}' needs left and right")
        if self.compare == "sign" and self.value is None:
            raise ValueError("truth compare 'sign' needs value")
        return self

    def refs(self) -> list[RefSpec]:
        """Every reference this truth expression uses."""
        return [r for r in (self.left, self.right, self.value) if r is not None]


class EvalCase(BaseModel):
    """One eval case: question, graders, and what each grader checks against."""

    model_config = ConfigDict(extra="forbid")
    id: str
    category: Category
    question: str
    required_tools: list[str] = []
    graders: list[GraderName]
    expected: list[RefSpec] = []
    rubric: str | None = None
    truth: TruthSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> EvalCase:
        if "judge" in self.graders and not self.rubric:
            raise ValueError("'judge' grader requires a rubric")
        if ("numeric" in self.graders or "date" in self.graders) and not self.expected:
            raise ValueError("'numeric' and 'date' graders require expected values")
        if self.rubric and "{truth}" in self.rubric and self.truth is None:
            raise ValueError("rubric uses {truth} but no truth is given")
        if "tools" in self.graders and not self.required_tools:
            raise ValueError("'tools' grader requires required_tools")
        return self


def load_cases(cases_dir: Path) -> list[EvalCase]:
    """Load every *.yaml, validate, and check cross-file rules:
    unique ids; every reference name in references.REGISTRY; every
    required_tools entry in agent.tools.TOOLS. Sorted by id.
    Raises ValueError listing ALL problems at once (not just the first).

    Why fail fast: a typo in a case should surface before any API money is
    spent, not halfway through a run."""
    from vixagent.agent.tools import TOOLS
    from vixagent.evals.references import REGISTRY

    problems: list[str] = []
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.yaml")):
        try:
            cases.append(EvalCase.model_validate(yaml.safe_load(path.read_text())))
        except (ValidationError, yaml.YAMLError) as exc:
            problems.append(f"{path.name}: {exc}")
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            problems.append(f"duplicate case id '{case.id}'")
        seen.add(case.id)
        refs = list(case.expected) + (case.truth.refs() if case.truth else [])
        for ref in refs:
            if ref.reference not in REGISTRY:
                problems.append(f"{case.id}: unknown reference '{ref.reference}'")
        for tool in case.required_tools:
            if tool not in TOOLS:
                problems.append(f"{case.id}: unknown tool '{tool}'")
    if problems:
        raise ValueError("Invalid eval cases:\n" + "\n".join(f"- {p}" for p in problems))
    return sorted(cases, key=lambda c: c.id)
