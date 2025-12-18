"""
This script helps you run McNemar's test for responses of two models.
"""

import dataclasses
import json
import math
import pathlib
from typing import Any

import polars as pl
import pydantic_core
import simple_parsing
from scipy.stats import binom, chi2


@dataclasses.dataclass(kw_only=True)
class Args(simple_parsing.Serializable):
    """
    Run McNemar's significance test for responses of two models.

    Samples must appear in the same order in both data frames.
    """

    df1: pathlib.Path
    """Parquet data frame with responses for model 1. Column: `response`."""
    df2: pathlib.Path
    """Parquet data frame with responses for model 2. Column: `response`."""


def parse_response(s: str) -> dict[str, Any] | None:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return pydantic_core.from_json(s, allow_partial=True)
    except ValueError:
        return None


def compute_accuracies(df: pl.DataFrame) -> dict[str, float]:
    assert df["language"].null_count() == 0, "Found missing language labels"
    assert df["label"].null_count() == 0, "Found missing reference tags"

    return {
        "language": len(df.filter(pl.col("parsed_language") == pl.col("language"))) / len(df),
        "label": len(df.filter(pl.col("parsed_label") == pl.col("label"))) / len(df),
        "null_language": df["parsed_language"].null_count(),
        "null_label": df["parsed_label"].null_count(),
    }


def mcnemar(res1: list[bool], res2: list[bool]) -> dict[str, float | int]:
    """Run McNemar's test for statistical significance."""
    if len(res1) != len(res2):
        raise ValueError(f"Sequences must have the same length, not {len(res1)} and {len(res2)}")

    # Formulas from here: https://en.wikipedia.org/wiki/McNemar%27s_test#Definition
    a = sum(1 for r1, r2 in zip(res1, res2) if r1 and r2)
    b = sum(1 for r1, r2 in zip(res1, res2) if r1 and (not r2))
    c = sum(1 for r1, r2 in zip(res1, res2) if (not r1) and r2)
    d = sum(1 for r1, r2 in zip(res1, res2) if (not r1) and (not r2))

    chi_square = (b - c) ** 2 / (b + c) if (b + c) != 0 else math.nan

    # Adapted from https://en.wikipedia.org/wiki/McNemar%27s_test#Example_implementation_in_Python
    if b + c >= 25:
        pvalue = chi2.sf(chi_square, 1)
    else:
        # In this case, chi_square is not well approximated by the chi-squared distribution.
        chi_square = math.nan
        n_min, n_max = min(b, c), max(b, c)
        pvalue = 2 * binom.cdf(n_min, n_min + n_max, 0.5) - binom.pmf(n_min, n_min + n_max, 0.5)
    assert isinstance(pvalue, float)

    return {"a": a, "b": b, "c": c, "d": d, "chi_square": chi_square, "pvalue": pvalue}


def all_mcnemars(df1: pl.DataFrame, df2: pl.DataFrame) -> dict[str, dict[str, int | float]]:
    def one_mcnemar(
        df1: pl.DataFrame, df2: pl.DataFrame, column: str, positive_value: Any
    ) -> dict[str, int | float]:
        """Run McNemar's test for values in a given column. Values are made binary using `positive_value`."""
        res1 = []
        res2 = []
        nulls = 0
        for v1, v2 in zip(df1[column], df2[column]):
            if v1 is None or v2 is None:
                nulls += 1
                continue
            assert isinstance(v1, type(positive_value)), f"Different types {v1=} vs {positive_value=}"
            assert isinstance(v2, type(positive_value)), f"Different types {v2=} vs {positive_value=}"
            res1.append(v1 == positive_value)
            res2.append(v2 == positive_value)
        return mcnemar(res1, res2) | {"nulls": nulls, "support": len(res1)}

    return {
        "label": one_mcnemar(df1, df2, "parsed_label", positive_value=1),
        **{
            lang: one_mcnemar(
                df1.filter(language=lang),
                df2.filter(language=lang),
                "parsed_language",
                positive_value=lang,
            )
            for lang in set(df1["language"].to_list() + df2["language"].to_list())
        },
    }


def main(args: Args) -> None:
    df1 = pl.read_parquet(args.df1)
    df2 = pl.read_parquet(args.df2)

    df1 = df1.with_columns(parsed_response=pl.Series([parse_response(r) for r in df1["response"]]))
    df2 = df2.with_columns(parsed_response=pl.Series([parse_response(r) for r in df2["response"]]))
    df1 = df1.with_columns(pl.col("parsed_response").name.prefix_fields("parsed_")).unnest("parsed_response")
    df2 = df2.with_columns(pl.col("parsed_response").name.prefix_fields("parsed_")).unnest("parsed_response")

    print(f"DF1: {compute_accuracies(df1)}")
    print(f"DF2: {compute_accuracies(df2)}")
    print()

    res = all_mcnemars(df1, df2)
    with pl.Config(tbl_width_chars=200, tbl_cols=100, tbl_hide_dataframe_shape=True):
        print("`b` and `c` count disagreements between the two models.")
        print(pl.DataFrame([{"binary test": k} | v for k, v in res.items()]))
        print()

    print("Compare against common significance levels:")
    for significance_level in [0.05, 0.005, 0.0005]:
        chi2_threshold = chi2.ppf(1 - significance_level, 1)
        print(f"Chi^2 Threshold for {significance_level:.2%} significance: {chi2_threshold}")


if __name__ == "__main__":
    main(Args.loads(simple_parsing.parse(Args, add_config_path_arg=True).dumps()))
