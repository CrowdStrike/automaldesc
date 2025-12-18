"""
This script lets you train an LLM using LoRA.
"""

import dataclasses
import datetime
import functools
import os
import pathlib
import random
import sys
import textwrap
from dataclasses import field
from typing import Any

import accelerate
import datasets
import jinja2
import mlflow
import numpy as np
import peft
import torch
import trl
from transformers import hf_argparser, modeling_utils, tokenization_utils
from transformers.models.auto import modeling_auto, tokenization_auto

from automaldesc import util


@dataclasses.dataclass
class OurArguments:
    model: str = field(metadata={"help": "Path to pretrained model or model identifier from HF"})
    train_dataset: str = field(metadata={"help": "Path to a Parquet file with training data"})
    test_dataset: str = field(metadata={"help": "Path to a Parquet file with test data"})

    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_r: int = field(default=8)
    lora_target_modules: str = field(
        default="gate_proj,down_proj,up_proj",
        metadata={"help": "Comma separated list of target modules to apply LoRA layers to. or `all-linear`"},
    )

    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Field used when loading the model. E.g., `eager` or `flash_attention_2`"},
    )
    use_reentrant: bool | None = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )

    # MLflow params.
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    mlflow_run_name: str | None = None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_and_prepare_model(
    args: OurArguments,
) -> tuple[modeling_utils.PreTrainedModel, peft.LoraConfig, tokenization_utils.PreTrainedTokenizer]:
    model = modeling_auto.AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=None,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
    )

    peft_config = peft.LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules
        ),
    )

    tokenizer = tokenization_auto.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        model, tokenizer = trl.setup_chat_format(model, tokenizer)

    return model, peft_config, tokenizer


def get_preprocessed_dataset(path: str, split: str) -> datasets.Dataset:
    prompt_input = pathlib.Path("./prompts/training.jinja").read_text(encoding="utf-8")
    prompt_answer = textwrap.dedent(
        """\
        {{ {
            "language": language,
            "label": label,
            "summary": summary
        } | json_dumps(indent=4) }}
        """
    )

    dataset = datasets.load_dataset("parquet", data_files={split: path}, split=split)
    assert isinstance(dataset, datasets.Dataset)

    def _reorient_hf_batch(batch: Any) -> list[dict[str, Any]]:
        """HF batches are column-oriented. Row-orient a given batch."""
        columns = batch.keys()
        rows = [dict(zip(columns, row_values)) for row_values in zip(*(batch[col] for col in columns))]
        return rows

    def apply_prompt_template(batch: Any, env: jinja2.Environment) -> dict[str, list[list[dict]]]:
        # Jinja templates cannot be pickled, so we recreate them every time for multiprocessing.
        template_input = env.from_string(prompt_input)
        template_answer = env.from_string(prompt_answer)
        rows = _reorient_hf_batch(batch)
        return {
            "messages": [
                [
                    {"role": "user", "content": template_input.render(**row)},
                    {"role": "assistant", "content": template_answer.render(**row)},
                ]
                for row in rows
            ]
        }

    # This is prompt formatting. Ideally done in parallel.
    dataset = dataset.map(
        functools.partial(apply_prompt_template, env=util.custom_jinja_env()),
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=min(24, cpus) if (cpus := os.cpu_count()) else 24,
        desc=f"Applying prompt template for {split}",
    )

    return dataset


def train(our_args: OurArguments, training_args: trl.SFTConfig) -> None:
    # Call `PartialState` with our timeout to initialize it correctly. Subsequent calls shouldn't need this.
    accelerate.PartialState(timeout=datetime.timedelta(seconds=training_args.ddp_timeout or 24 * 60 * 60))
    # Using a barrier here might help timeouts: https://github.com/pytorch/pytorch/issues/124950.
    accelerate.PartialState().wait_for_everyone()

    set_seed(training_args.seed)

    model, peft_config, tokenizer = create_and_prepare_model(our_args)

    if not accelerate.PartialState().is_main_process:
        print("[automaldesc] Waiting for the main process to prepare the dataset first")
    with accelerate.PartialState().main_process_first():
        train_dataset = get_preprocessed_dataset(our_args.train_dataset, split="train")
        test_dataset = get_preprocessed_dataset(our_args.test_dataset, split="test")
        print(f"[automaldesc] Training dataset has {len(train_dataset)} samples")
        print(f"[automaldesc] Test dataset has {len(test_dataset)} samples")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": our_args.use_reentrant}
    trainer = trl.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
    )
    if trainer.accelerator.is_main_process:
        trainer.accelerator.print(f"[automaldesc] {our_args=}")
        trainer.accelerator.print(f"[automaldesc] {training_args=}")
        trainer.accelerator.print(f"{trainer.model}")
        if hasattr(trainer.model, "print_trainable_parameters"):
            trainer.model.print_trainable_parameters()  # pyright: ignore

        if our_args.mlflow_tracking_uri:
            trainer.accelerator.print("[automaldesc] Setting up MLflow run")
            os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
            mlflow.enable_system_metrics_logging()
            mlflow.set_tracking_uri(our_args.mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=our_args.mlflow_experiment)
            mlflow.set_tag("mlflow.runName", our_args.mlflow_run_name)
            mlflow.log_dict(dataclasses.asdict(our_args), "our_args.json")
            mlflow.log_dict(dataclasses.asdict(training_args), "training_args.json")

    # Train the model.
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(f"{training_args.output_dir}/tokenizer")
    print(f"[automaldesc] Going to train ({trainer.accelerator.process_index})")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final model.
    trainer.save_model()
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(f"{training_args.output_dir}/tokenizer")


def main() -> None:
    parser = hf_argparser.HfArgumentParser((OurArguments, trl.SFTConfig))  # type: ignore
    # If we pass only one argument to the script, and it seems to be a file path, parse the file.
    # Else, parse the CLI arguments.
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # pylint: disable-next=unbalanced-tuple-unpacking
        our_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        # pylint: disable-next=unbalanced-tuple-unpacking
        our_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        # pylint: disable-next=unbalanced-tuple-unpacking
        our_args, training_args = parser.parse_args_into_dataclasses()

    train(our_args, training_args)


if __name__ == "__main__":
    main()
