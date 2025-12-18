"""
This script helps you evaluate the responses of a model.
"""

import dataclasses
import json
import pathlib
from collections.abc import Collection
from typing import Any, Iterable

import polars as pl
import pydantic
import pydantic_core
import simple_parsing

from automaldesc import util


@dataclasses.dataclass(kw_only=True)
class Args(simple_parsing.Serializable):
    """
    Parse candidate responses and evaluate them against a reference set.

    Samples must appear in the same order both datasets.
    """

    predictions: pathlib.Path
    """Data frame with raw candidate responses, as strings. Columns: `response`."""
    references: pathlib.Path
    """Data frame with reference metadata. Columns: `label`, `language`, `summary`."""


class AutoMalDescScore:
    """Custom metric for AutoMalDesc."""

    @pydantic.dataclasses.dataclass(kw_only=True)
    class Response:
        """Schema expected from candidate responses."""

        language: str
        label: int
        summary: str

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the metric to its initial state. Use this to clear previous predictions."""
        self.references: list[AutoMalDescScore.Response] = []
        self.candidates: list[AutoMalDescScore.Response | None] = []

    def add_batch(self, *, predictions: Iterable[str], references: Iterable[str]) -> None:
        """Account for one more sequence of predictions."""
        for p, r in zip(predictions, references):
            self.add_one(p, r)

    def add_one(self, prediction: str, reference: str) -> None:
        """Account for one more prediction."""
        try:
            self.references.append(pydantic.TypeAdapter(AutoMalDescScore.Response).validate_json(reference))
        except pydantic_core.ValidationError as e:
            raise ValueError("Failed to parse reference JSON") from e

        self.candidates.append(self._parse_candidate_json(prediction))

    def _parse_candidate_json(self, s: str) -> Response | None:
        try:
            candidate = pydantic_core.from_json(s, allow_partial=True)
            candidate = pydantic.TypeAdapter(AutoMalDescScore.Response).validate_python(candidate)
            return candidate
        except (ValueError, AssertionError):
            return None

    def compute(self) -> dict[str, float]:
        """Process and return the scores."""

        def accuracy(can: Collection[Any | None], ref: Collection[Any]) -> float:
            correct = sum(1 for c, r in zip(can, ref) if c is not None and r == c)
            total = len(ref)
            return correct / total

        df = pl.DataFrame(
            {
                "ref_language": [r.language for r in self.references],
                "ref_label": [r.label for r in self.references],
                "ref_summary": [r.summary for r in self.references],
                "can_language": [c.language if c is not None else None for c in self.candidates],
                "can_label": [c.label if c is not None else None for c in self.candidates],
                "can_summary": [c.summary if c is not None else None for c in self.candidates],
            }
        )

        return {
            "json-err-fraction": sum(1 for c in self.candidates if c is None) / len(self.candidates),
            "acc-language": accuracy(df["can_language"], df["ref_language"]),
            "acc-label": accuracy(df["can_label"], df["ref_label"]),
            "acc-summary": accuracy(df["can_summary"], df["ref_summary"]),
            **{
                k: v
                for (lang,), group in sorted(df.group_by("ref_language"))
                for k, v in {
                    f"acc-{lang}-language": accuracy(group["can_language"], group["ref_language"]),
                    f"acc-{lang}-label": accuracy(group["can_label"], group["ref_label"]),
                    f"acc-{lang}-summary": accuracy(group["can_summary"], group["ref_summary"]),
                }.items()
            },
        }


def main(args: Args) -> None:
    predictions = util.smart_polars_read(args.predictions)
    references = util.smart_polars_read(args.references)

    metric = AutoMalDescScore()
    metric.add_batch(
        predictions=predictions["response"].to_list(),
        references=[
            json.dumps({"language": row["language"], "label": row["label"], "summary": row["summary"]})
            for row in references.iter_rows(named=True)
        ],
    )
    print(metric.compute())


if __name__ == "__main__":
    main(Args.loads(simple_parsing.parse(Args, add_config_path_arg=True).dumps()))
