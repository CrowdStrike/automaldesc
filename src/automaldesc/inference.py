"""
This script lets you run inference jobs with vLLM.
"""

import dataclasses
import json
import logging
import pathlib
from typing import Any

import jinja2
import polars as pl
import simple_parsing
import tqdm
import vllm
import vllm.lora.request

from automaldesc import util


@dataclasses.dataclass
class EngineParams:
    """vLLM `EngineArgs`."""

    tensor_parallel_size: int = 1
    enable_chunked_prefill: bool = False
    max_model_len: int | None = None


@dataclasses.dataclass
class SamplingParams:
    """vLLM `SamplingParams`."""

    temperature: float = 0.0
    max_tokens: int = 1024
    logprobs: int | None = None
    """Number of log probabilities to return per token. Logprobs are saved with custom formatting."""


@dataclasses.dataclass(kw_only=True)
class Args(simple_parsing.Serializable):
    """Run inference with a given LLM and prompt."""

    inputs: pathlib.Path
    """Data frame with input metadata."""
    prompt: pathlib.Path
    """Path to the Jinja prompt."""
    system: pathlib.Path | None = None
    """Path to the system prompt file."""

    joblog: pathlib.Path
    """Data frame with previous job statuses (can be a missing file)."""
    retry_failed: bool = False
    """Whether to retry failed jobs from the joblog."""

    # Outputs.
    out_frame: pathlib.Path
    """JSONL Dataframe where to store outputs."""
    out_log: pathlib.Path = pathlib.Path("log.txt")
    """File where to log output messages."""
    out_joblog: pathlib.Path = pathlib.Path("joblog.jsonl")
    """JSONL dataframe to be used as a joblog (storing job status for resuming)."""

    # Inference parameters.
    model: str
    """Path or model identifier to use. For LoRA inference, this is the base model."""
    lora: str | None = None
    """Path to LoRA adapters to use, if needed."""
    tokenizer: str | None = None
    """Path to the tokenizer, if different from the model."""
    chat_template: bool = False
    """Whether to add a chat template to the prompt."""
    bs: int = 1
    """Number of prompts to give to vLLM at once."""
    engine_params: EngineParams
    sampling_params: SamplingParams


def prepare_jobs(args: Args) -> pl.DataFrame:
    df = util.smart_polars_read(args.inputs).with_row_index()
    logging.info("Read %d jobs", len(df))

    # Filter unnecessary jobs.
    if args.joblog.is_file() and args.joblog.stat().st_size > 0:
        joblog = pl.read_ndjson(args.joblog).unique("index", keep="last")
        if args.retry_failed:
            df = df.filter(~pl.col("index").is_in(set(joblog.filter(status="ok")["index"])))
        else:
            df = df.filter(~pl.col("index").is_in(set(joblog["index"])))
        logging.info("Filtered down to %d jobs", len(df))

    return df


def prepare_template(prompt_path: pathlib.Path) -> jinja2.Template:
    return util.custom_jinja_env().from_string(prompt_path.read_text(encoding="utf-8"))


def main(args: Args) -> None:
    logging.info("Running with %s", args.dumps_json(indent=4))

    df = prepare_jobs(args)
    if df.is_empty():
        logging.info("Don't have any jobs to do. Exiting")
        return

    template = prepare_template(args.prompt)

    system_prompt = args.system.read_text() if args.system else None
    if args.system and not args.chat_template:
        raise ValueError("If you want to set a system prompt, you need to activate chat templating.")

    sampling_params = vllm.SamplingParams(**dataclasses.asdict(args.sampling_params))
    llm = vllm.LLM(
        args.model,
        tokenizer=args.tokenizer,
        enable_lora=(args.lora is not None),
        **dataclasses.asdict(args.engine_params),
    )
    logging.info("Loaded model and vLLM parameters")

    if args.chat_template and getattr(llm.get_tokenizer(), "chat_template", None) is None:
        raise ValueError("You requested adding a chat template, but the tokenizer does not have one.")

    def format_prompt(metadata: dict[str, Any]) -> str:
        prompt = template.render(**metadata)

        if args.chat_template:
            messages = [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ]
            prompt = str(
                llm.get_tokenizer().apply_chat_template(  # type: ignore
                    messages, tokenize=False, add_generation_prompt=True  # type: ignore
                )
            )

        return prompt

    with (
        open(args.out_frame, "a", encoding="utf-8", buffering=1) as fframe,
        open(args.out_joblog, "a", encoding="utf-8", buffering=1) as fjoblog,
    ):
        for chunk in tqdm.tqdm(list(df.iter_slices(args.bs)), desc="Processing batches"):
            prompts = [format_prompt(row) for row in chunk.iter_rows(named=True)]

            try:
                outputs = llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    lora_request=(
                        None if args.lora is None else vllm.lora.request.LoRARequest("adapter", 1, args.lora)
                    ),
                )
            except (RuntimeError, ValueError) as e:
                logging.warning("Failed to run batch. Marking jobs as failed. Reason: %s", e)
                for row in chunk.iter_rows(named=True):
                    fjoblog.write(json.dumps({"index": row["index"], "status": "failed"}) + "\n")
                continue

            for output, row in zip(outputs, chunk.iter_rows(named=True)):
                res = {"index": row["index"], "response": output.outputs[0].text}
                try:
                    out = output.outputs[0]
                    if (logprobs := getattr(out, "logprobs", None)) is not None:
                        # Convert logprobs into a serializable form.
                        res["logprobs"] = [{k: dataclasses.asdict(v) for k, v in d.items()} for d in logprobs]
                    if hasattr(out, "token_ids"):
                        res["token_ids"] = [int(tid) for tid in out.token_ids]
                except (ValueError, RuntimeError) as e:
                    logging.error("Error processing logprobs: %s", e)
                    res["logprobs_err"] = str(e)

                fframe.write(json.dumps(res) + "\n")
                fjoblog.write(json.dumps({"index": row["index"], "status": "ok"}) + "\n")

    logging.info("Done. Exiting")


if __name__ == "__main__":
    # Call `dumps` and `loads` in case nested objects don't have the right type.
    main(Args.loads(simple_parsing.parse(Args, add_config_path_arg=True).dumps()))
