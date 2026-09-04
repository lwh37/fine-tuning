#!/usr/bin/env python3
"""Public-safe E1-reference / E2-LoRA Vanilla GRPO training entry point.

The script contains no private paths or data. It mirrors the selected experiment
design; compatibility may require minor updates across TRL/PEFT releases.
"""

import argparse
import json
import types
from contextlib import contextmanager
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo_reward import binary_reward


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return ""


def reward_function(completions, binary_label, **kwargs):
    return [binary_reward(completion_text(value), gt) for value, gt in zip(completions, binary_label)]


reward_function.__name__ = "uniform_binary_reward"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--e1-adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "binary_label": int(row["binary_label"]),
            "bucket": row["bucket"],
            "sample_id": row["sample_id"],
        }
        for row in rows
    ])
    tokenizer = AutoTokenizer.from_pretrained(args.e1_adapter, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    e2_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = PeftModel.from_pretrained(base, args.e1_adapter, adapter_name="e1", is_trainable=False)
    model.add_adapter("e2", e2_config)
    model.base_model.set_adapter(["e1", "e2"], inference_mode=False)

    def freeze_e1_train_e2(current):
        for name, parameter in current.named_parameters():
            parameter.requires_grad = ".e2." in name

    freeze_e1_train_e2(model)

    @contextmanager
    def e1_reference_context(self):
        self.base_model.set_adapter("e1", inference_mode=True)
        try:
            yield
        finally:
            self.base_model.set_adapter(["e1", "e2"], inference_mode=False)
            freeze_e1_train_e2(self)

    # When ref_model=None, GRPOTrainer calls disable_adapter(). Override the
    # context so the reference is Base+E1 rather than the original Base.
    model.disable_adapter = types.MethodType(e1_reference_context, model)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(".e2." not in name for name in trainable):
        raise RuntimeError("only the E2 adapter may be trainable")

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        generation_batch_size=8,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        disable_dropout=True,
        max_prompt_length=1024,
        num_generations=8,
        max_completion_length=10,
        temperature=0.9,
        top_p=1.0,
        beta=0.01,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        seed=20260830,
        data_seed=20260830,
        logging_steps=25,
        save_strategy="steps",
        save_steps=128,
        report_to="tensorboard",
        log_completions=False,
        ddp_find_unused_parameters=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
