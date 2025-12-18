"""
This module contains generally-useful code that does not belong somewher else.
"""

import json
import pathlib
from typing import Any

import jinja2
import polars as pl


def smart_polars_read(path: pathlib.Path) -> pl.DataFrame:
    """Call the regular Polars read function with the format suggested by the file extension."""
    match path.suffix:
        case ".jsonl" | ".ndjson":
            return pl.read_ndjson(path)
        case ".json":
            return pl.read_json(path)
        case ".parquet":
            return pl.read_parquet(path)
        case ".csv":
            return pl.read_csv(path)
        case ".xls" | ".xlsx" | ".xlsb":
            return pl.read_excel(path)
        case ".avro":
            return pl.read_avro(path)
    raise ValueError(f"Don't know how to read the '{path.suffix}' format (file extension)")


def custom_jinja_env() -> jinja2.Environment:
    """Custom Jinja environment, with extra capabilities (filters)."""

    def read_file(value: str, encoding: str = "utf-8") -> str:
        """Read the contents of the specified file. Meant to be used as a Jinja filter."""
        return pathlib.Path(value).read_text(encoding=encoding, errors="replace")

    def json_dumps(value: Any, **kwargs) -> str:
        """Simply run the regular `json.dumps` function. Slight differences from Jinja's `tojson`."""
        return json.dumps(value, **kwargs)

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, extensions=["jinja2.ext.do"])
    env.filters["read_file"] = read_file
    env.filters["json_dumps"] = json_dumps
    return env
