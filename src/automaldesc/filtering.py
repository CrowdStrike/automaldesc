"""
This script runs the filtering pipeline over the given dataframes.
"""

import dataclasses
import json
import multiprocessing
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pydantic_core
import simple_parsing
import tqdm


@dataclasses.dataclass(kw_only=True)
class Args(simple_parsing.Serializable):
    """Run the filtering pipeline."""

    dfs: list[pathlib.Path]
    """JSONL files with inference responses, log probabilities and coherence responses from an LLM judge."""
    out_dir: pathlib.Path
    """Directory where the processed dataframes get saved."""

    confidence_threshold: float = 0.90
    """Threshold for confidence filtering. Number between 0 and 1."""
    workers: int = 64
    """Number of workers to use to parallelise data loading."""


def read_df(path: pathlib.Path, workers: int | None = None) -> pd.DataFrame:
    print(f"Reading {path}")
    lines = path.read_text().splitlines()

    # Process data in parallel, to deal with a large number of logprobs.
    with multiprocessing.Pool(workers) as pool:
        data = tqdm.tqdm(
            pool.imap_unordered(json.loads, lines, chunksize=200), total=len(lines), desc="Parsing JSON"
        )
        return pd.DataFrame(list(data)).sort_values("sha256")


def remove_empty_responses(df: pd.DataFrame) -> pd.DataFrame:
    empty_responses = df[df["response"] == ""]
    print(f"Total empty responses: {len(empty_responses)}")

    empty_distribution = pd.crosstab(empty_responses["language"], empty_responses["label"])

    print("\nPercentage of empty responses by language and label:")
    for lang in empty_distribution.index:
        for label in empty_distribution.columns:
            total = df[(df["language"] == lang) & (df["label"] == label)].shape[0]
            empty = empty_distribution.loc[lang, label]
            if total > 0:
                percentage = (empty / total) * 100
                print(f"{lang}, Label {label}: {empty}/{total} ({percentage:.2f}%)")

    df_filtered = df[df["response"] != ""]

    if "__index_level_0__" in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=["__index_level_0__"])

    print(f"\nOriginal count: {len(df)}")
    print(f"After removing empty responses: {len(df_filtered)}")
    print(f"Removed {len(df) - len(df_filtered)} empty responses")

    return df_filtered


def filter_invalid_json(df: pd.DataFrame) -> pd.DataFrame:
    extracted_data = []
    truncated_count = 0
    empty_count = 0

    for _, row in df.iterrows():
        # Fail if this is not JSON.
        try:
            pydantic_core.from_json(row["response"], allow_partial=True)
        except ValueError:
            continue

        try:
            response = row["response"]

            if '"summary":' not in response or '"label":' not in response or '"language":' not in response:
                empty_count += 1
                continue

            summary_start = response.find('"summary":') + len('"summary":')
            summary_end = response.find(',"', summary_start)
            if summary_end == -1:  # if summary is the last field
                summary_end = response.find("}", summary_start)

            summary = response[summary_start:summary_end].strip().strip('"')

            if not summary or summary.endswith("..."):
                truncated_count += 1
                continue
            if not summary.strip():
                empty_count += 1
                continue

            if (
                '""' in response.split('"label":')[1].split(",")[0]
                or '""' in response.split('"language":')[1].split(",")[0]
            ):
                empty_count += 1
                continue

            extracted_data.append(row)

        except (ValueError, RuntimeError) as e:
            print(f"Unexpected error processing row: {str(e)}")
            continue

    filtered_df = pd.DataFrame(extracted_data)

    print(f"Total responses processed: {len(df)}")
    print(f"Valid responses: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.2f}%)")
    print(f"Truncated summaries removed: {truncated_count}")
    print(f"Empty fields removed: {empty_count}")

    return filtered_df


def filter_label_mismatches(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    def get_label(x: Any) -> str | None:
        if isinstance(x, str):
            try:
                return json.loads(x)["label"]
            except (json.JSONDecodeError, KeyError):
                return None
        elif isinstance(x, dict):
            return x.get("label")
        return None

    label_tables = [{row["sha256"]: get_label(row["response"]) for _, row in df.iterrows()} for df in dfs]
    consistent_shas = set(
        sha for sha, label in label_tables[0].items() if all(t.get(sha, None) == label for t in label_tables)
    )

    print(f"Original sizes: {list(map(len, dfs))}")
    dfs = [df[df["sha256"].isin(consistent_shas)] for df in dfs]
    print(f"Filtered sizes: {list(map(len, dfs))}")
    return dfs


def find_lobprob_record(
    key: str, logprobs: list[dict[str, dict[str, Any]]], token_ids: list[int], verbose: bool = False
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Find the logprob information of the value corresponding to a given JSON key."""

    assert len(logprobs) == len(token_ids), f"{len(logprobs)=} {len(token_ids)=}"

    tokens = [logp[str(tid)]["decoded_token"] for logp, tid in zip(logprobs, token_ids)]

    # Binary search for the first position which contains our value.
    # For string values, we search for the beginning of the string.
    offset = 0
    power = 1 << 20  # Should be larger than the number of tokens in the response. Power of two!
    while power > 0:
        maybe_offset = offset + power
        power >>= 1
        if maybe_offset > len(tokens):
            continue
        try:
            s = "".join(tokens[:maybe_offset]) + '"'  # Try to close strings ourselves.
            obj = pydantic_core.from_json(s, allow_partial=True)
        except (RuntimeError, ValueError):
            obj = {}
        if key not in obj:
            offset = maybe_offset

    # For string values, offset might point to an opening quotation mark, so move it.
    while offset < len(tokens) and tokens[offset].strip() in ['"', ""]:
        offset += 1

    # Some inputs fail with binary search. Attempt linear search for them.
    if offset < 0 or offset >= len(token_ids):
        for offset in range(len(token_ids)):
            try:
                s = "".join(tokens[:offset]) + '"'  # Try to close strings ourselves.
                obj = pydantic_core.from_json(s, allow_partial=True)
            except (RuntimeError, ValueError):
                obj = {}
            if key in obj:
                offset -= 1
                break

        # For string values, offset might point to an opening quotation mark, so move it.
        while offset < len(tokens) and tokens[offset].strip() in ['"', ""]:
            offset += 1

    if verbose:
        if offset < 0 or offset >= len(logprobs):
            tqdm.tqdm.write(f"{key=} {offset=} {len(logprobs)=}")
        tqdm.tqdm.write(f"Considering the prefix '{tokens[:offset+1]}'")
    return offset, logprobs[offset]


def extract_specific_logprobs(df: pd.DataFrame) -> pd.DataFrame:
    new_data = []
    errors = 0

    for _, row in (pbar := tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting logprobs")):
        try:
            tid_lang, logprob_lang = find_lobprob_record("language", row["logprobs"], row["token_ids"])
            tid_label, logprob_label = find_lobprob_record("label", row["logprobs"], row["token_ids"])
            new_row = {
                "sha256": row["sha256"],
                "path": row["path"],
                "coherence_response": row["coherence_response"],
                "response": row["response"],
                "token_id_language": tid_lang,
                "token_id_label": tid_label,
                "logprobs_language": logprob_lang,
                "logprobs_label": logprob_label,
            }
            new_data.append(new_row)

        except (ValueError, RuntimeError) as e:
            pbar.write(f"Error processing row with SHA256 {row['sha256']}: {str(e)}")
            errors += 1
            pbar.set_description(f"Extracting logprobs ({errors} errors)")
            continue

    return pd.DataFrame(new_data)


def convert_logprob_to_prob(logprob: float) -> float:
    return np.exp(logprob)


def get_first_logprob(logprobs_dict) -> float | None:
    """Extract the first logprob value from the nested dictionary"""
    try:
        first_token = list(logprobs_dict.keys())[0]
        return logprobs_dict[first_token]["logprob"]
    except (RuntimeError, ValueError):
        return None


def filter_by_confidence(
    df: pd.DataFrame, threshold: float, fields_to_filter: tuple[str, ...] = ("label",)
) -> pd.DataFrame:
    """Filter dataframe keeping only samples equal to or above a confidence threshold."""

    filtered_df = df.copy()

    if "label" in fields_to_filter:
        print("Filtering by label confidence")
        label_probs = filtered_df["logprobs_label"].apply(get_first_logprob).apply(convert_logprob_to_prob)
        filtered_df = filtered_df[label_probs >= threshold]

    if "language" in fields_to_filter:
        print("Filtering by language confidence")
        lang_probs = filtered_df["logprobs_language"].apply(get_first_logprob).apply(convert_logprob_to_prob)
        filtered_df = filtered_df[lang_probs >= threshold]

    return filtered_df


def filter_by_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe keeping only samples where the summary is coherent with the predicted label."""

    def get_label(x: Any) -> str | None:
        if isinstance(x, str):
            try:
                return json.loads(x)["label"]
            except (json.JSONDecodeError, KeyError):
                return None
        elif isinstance(x, dict):
            return x.get("label")
        return None

    mask: list[bool] = []
    for _, row in df.iterrows():
        try:
            summary_label = int(row["coherence_response"])
            mask.append(summary_label == get_label(row["response"]))
        except TypeError:
            mask.append(False)

    df = df.filter(mask)

    return df


def prepare_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def extract_predictions(response: str) -> tuple[int, str, str] | tuple[None, None, None]:
        try:
            parsed = json.loads(response)
            return int(parsed["label"]), parsed["language"].lower(), parsed["summary"]
        except (ValueError, RuntimeError):
            return None, None, None

    df[["predicted_label", "predicted_language", "predicted_summary"]] = pd.DataFrame(
        df["response"].apply(extract_predictions).tolist(), index=df.index
    )

    allowed_languages = {"javascript", "batch", "bash", "python", "powershell"}
    if (bad_mask := ~df["predicted_language"].isin(allowed_languages)).any():
        print(f"Found unexpected languages in predictions: {df[bad_mask]}")
        df = df[~bad_mask]

    final_data = [
        {
            "sha256": row["sha256"],
            "path": row["path"],
            "label": int(row["predicted_label"]),
            "language": row["predicted_language"],
            "summary": row["predicted_summary"],
        }
        for _, row in df.iterrows()
    ]
    return pd.DataFrame(final_data)


def main(args: Args) -> None:
    dfs = [read_df(p, args.workers) for p in args.dfs]
    print("Removing empty responses")
    dfs = [remove_empty_responses(df) for df in dfs]
    print("Removing parsing errors")
    dfs = [filter_invalid_json(df) for df in dfs]
    print("Removing label mismatches")
    dfs = filter_label_mismatches(dfs)
    print("Processing log probs")
    dfs = [extract_specific_logprobs(df) for df in dfs]
    print("Removing by confidence")
    dfs = [filter_by_confidence(df, args.confidence_threshold) for df in dfs]
    print("Removing by coherence")
    dfs = [filter_by_coherence(df) for df in dfs]

    print("Preparing for training")
    dfs = [prepare_for_training(df) for df in dfs]

    for path, df in zip(args.dfs, dfs):
        out_path = (args.out_dir / path.name).with_suffix(".parquet")
        print(f"Saving to {out_path}")
        df.to_parquet(out_path.as_posix())


if __name__ == "__main__":
    main(Args.loads(simple_parsing.parse(Args, add_config_path_arg=True).dumps()))
