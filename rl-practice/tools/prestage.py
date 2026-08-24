#!/usr/bin/env python3
"""Produce every artefact the notebooks can fall back on, so nothing depends on the day.

Assume nothing about student network access or patience. This script downloads the base
model once, runs the author's own reference training for both GRPO acts and for the four
DQN configs, and writes the results into ``assets/``. Every training cell in the notebooks
has a ``TRAIN_FROM_SCRATCH`` flag; when it is False the cell loads what this script
produced and every downstream plot and evaluation still works.

That is the insurance policy. If a student's GPU misbehaves, or a run diverges, or the
room simply runs out of time, they flip one flag and continue with the analysis rather
than losing the exercise. It is also how the notebooks stay runnable on a laptop.

Usage::

    python tools/prestage.py --all           # everything, roughly 45 minutes on a T4
    python tools/prestage.py --model         # just download the base weights (~1 GB)
    python tools/prestage.py --dqn           # just notebook 1 (CPU, a few minutes)
    python tools/prestage.py --act1 --act4   # just the GRPO reference runs

Artefacts, all under ``assets/``::

    base_model/                 the model weights, loaded with local_files_only=True
    reference_adapters/act1/    LoRA adapter from the beta = 0.0 run
    reference_adapters/act4/    LoRA adapter from the beta = 0.04 run
    reference_logs/act1.csv     per-step metrics from that run
    reference_logs/act4.csv
    snapshots/before.json       frozen evaluation of the base model
    snapshots/after_act1.json   frozen evaluation after the beta = 0.0 run
    snapshots/after_act4.json   frozen evaluation after the beta = 0.04 run
    dqn/{working,CONFIG_A,CONFIG_B,CONFIG_C}.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ASSETS = os.path.join(HERE, "assets")

# --- the geometry of the reference runs lives in rlpractice.grpo, so that the notebooks
# --- and this script cannot disagree about what was trained on.
from rlpractice.grpo import (  # noqa: E402
    ACC_EVERY,
    ACC_N,
    ACT1_TIME_BUDGET_SECONDS,
    ACT4_TIME_BUDGET_SECONDS,
    MAX_STEPS_CAP,
    N_EVAL,
    PPL_EVERY,
    TRAIN_MIX,
    build_splits,
)

def prestage_model(verbose: bool = True) -> str:
    """Download the base model to a known local path, once."""
    from huggingface_hub import snapshot_download

    from rlpractice.grpo import MODEL_ID

    target = os.path.join(ASSETS, "base_model")
    if os.path.exists(os.path.join(target, "config.json")):
        if verbose:
            print(f"[model] already present at {target}")
        return target

    if verbose:
        print(f"[model] downloading {MODEL_ID} -> {target}")
    snapshot_download(
        MODEL_ID,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "merges.txt", "vocab.json"],
    )
    if verbose:
        total = sum(
            os.path.getsize(os.path.join(target, f))
            for f in os.listdir(target)
            if os.path.isfile(os.path.join(target, f))
        )
        print(f"[model] {total / 1024**3:.2f} GB at {target}")
    return target


def run_act(
    tag: str,
    beta: float,
    time_budget_seconds: float,
    baseline: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run one reference GRPO act end to end and save adapter, log and snapshot."""
    import torch

    from rlpractice import evaluation, grpo
    from rlpractice.callbacks import (
        InstrumentedGRPOTrainer,
        MetricsCSVCallback,
        NaNGuardCallback,
        TimeBudgetCallback,
        peak_memory_report,
    )
    from rlpractice.rewards import correctness_reward, format_reward

    snapshots_dir = os.path.join(ASSETS, "snapshots")
    logs_dir = os.path.join(ASSETS, "reference_logs")
    adapter_dir = os.path.join(ASSETS, "reference_adapters", tag)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    train_dataset, eval_dataset = build_splits()
    model, tokenizer = grpo.load_model_and_tokenizer(verbose=verbose)

    # The "before" snapshot is shared by both acts: both start from the same base model,
    # so measuring it twice would be wasted time and an invitation to drift.
    if baseline is None:
        baseline_path = os.path.join(snapshots_dir, "before.json")
        if os.path.exists(baseline_path):
            baseline = evaluation.load_snapshot("before", snapshots_dir)
            if verbose:
                print(f"[{tag}] reusing baseline snapshot from {baseline_path}")
        else:
            if verbose:
                print(f"[{tag}] measuring the base model (frozen evaluation)...")
            baseline = evaluation.snapshot(
                model, tokenizer, eval_dataset, "before", snapshots_dir, n_task=N_EVAL
            )

    torch.cuda.reset_peak_memory_stats()
    csv_path = os.path.join(logs_dir, f"{tag}.csv")

    config = grpo.build_grpo_config(
        beta=beta,
        max_steps=MAX_STEPS_CAP,
        output_dir=os.path.join(ASSETS, "_trainer_output", tag),
        disable_tqdm=True,
    )
    if verbose:
        print(f"\n[{tag}] GRPO configuration")
        print(grpo.describe_config(config))

    budget = TimeBudgetCallback(time_budget_seconds, MAX_STEPS_CAP)
    metrics = MetricsCSVCallback(
        csv_path=csv_path,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        ppl_every=PPL_EVERY,
        acc_every=ACC_EVERY,
        acc_n=ACC_N,
        dashboard_every=None,          # no live plot when running headless
        baseline_ppl=baseline["general_perplexity"],
        baseline_accuracy=baseline["task_accuracy"],
        title=f"reference run: {tag} (beta={beta})",
    )

    trainer = InstrumentedGRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=config,
        train_dataset=train_dataset,
        peft_config=grpo.lora_config(),
        callbacks=[NaNGuardCallback(), budget, metrics],
    )

    started = time.time()
    trainer.train()
    wall = time.time() - started

    trainer.model.save_pretrained(adapter_dir)
    # Deliberately no tokenizer.save_pretrained here. It would be a byte-identical copy of
    # the base model's tokenizer, 15 MB per adapter, and nothing reads it: the notebooks
    # get their tokenizer from grpo.load_model_and_tokenizer, and PeftModel.from_pretrained
    # reads only adapter_config.json and adapter_model.safetensors.

    # Raw completions sampled during training. Act 3 prints these as a table so the drift
    # in what the model actually writes is legible; without them the fallback path can
    # plot curves but cannot show the thing the curves are about.
    with open(os.path.join(logs_dir, f"{tag}_completions.json"), "w") as handle:
        json.dump(
            {str(step): texts for step, texts in getattr(trainer, "completion_samples", {}).items()},
            handle,
            indent=2,
        )

    after = evaluation.snapshot(
        trainer.model, tokenizer, eval_dataset, f"after_{tag}", snapshots_dir, n_task=N_EVAL
    )
    memory = peak_memory_report()

    summary = {
        "tag": tag,
        "beta": beta,
        "steps": trainer.state.global_step,
        "seconds_per_step": budget.seconds_per_step,
        "budget_steps": budget.budget_steps,
        "wall_seconds": wall,
        "eval_seconds": metrics.eval_seconds,
        "peak_allocated_gb": memory["allocated_gb"],
        "peak_reserved_gb": memory["reserved_gb"],
        "baseline_accuracy": baseline["task_accuracy"],
        "final_accuracy": after["task_accuracy"],
        "baseline_perplexity": baseline["general_perplexity"],
        "final_perplexity": after["general_perplexity"],
        "adapter": adapter_dir,
        "log_csv": csv_path,
    }
    with open(os.path.join(logs_dir, f"{tag}_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    if verbose:
        print(f"\n[{tag}] done in {wall / 60:.1f} min over {summary['steps']} steps")
        print(f"  task accuracy      {summary['baseline_accuracy']:.3f} -> {summary['final_accuracy']:.3f}")
        print(
            f"  general perplexity {summary['baseline_perplexity']:.3f} -> "
            f"{summary['final_perplexity']:.3f} "
            f"(x{summary['final_perplexity'] / summary['baseline_perplexity']:.2f})"
        )
        print(f"  peak memory        {memory['allocated_gb']:.2f} GB allocated")

    del trainer, model
    torch.cuda.empty_cache()
    return summary


def prestage_dqn(verbose: bool = True) -> dict:
    """Run the four notebook-1 configurations and save weights plus reward curves."""
    from rlpractice import dqn

    target = os.path.join(ASSETS, "dqn")
    os.makedirs(target, exist_ok=True)
    summaries = {}
    for label, config in dqn.CONFIGS.items():
        path = os.path.join(target, f"{label}.pt")
        if os.path.exists(path):
            if verbose:
                print(f"[dqn] {label} already present")
            summaries[label] = dqn.load_result(path).summary()
            continue
        if verbose:
            print(f"[dqn] training {label} ...")
        env = dqn.make_env(config.env_id, seed=config.seed)
        result = dqn.train_dqn(env, config, label=label, verbose=verbose)
        env.close()
        dqn.save_result(result, path)
        summaries[label] = result.summary()
        if verbose:
            print(f"  {result.summary()}")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="model + dqn + act1 + act4")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--dqn", action="store_true")
    parser.add_argument("--act1", action="store_true")
    parser.add_argument("--act4", action="store_true")
    parser.add_argument("--act1-seconds", type=float, default=ACT1_TIME_BUDGET_SECONDS)
    parser.add_argument("--act4-seconds", type=float, default=ACT4_TIME_BUDGET_SECONDS)
    parser.add_argument("--lr", type=float, default=None, help="override the learning rate for both acts")
    parser.add_argument("--lora-r", type=int, default=None, help="override the LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="override the LoRA alpha")
    parser.add_argument("--suffix", default="", help="suffix appended to artefact tags")
    args = parser.parse_args(argv)

    if not any([args.all, args.model, args.dqn, args.act1, args.act4]):
        parser.print_help()
        return 1

    from rlpractice import grpo as _grpo

    if args.lr is not None:
        _grpo.GRPO_COMMON["learning_rate"] = args.lr
        print(f"[prestage] learning rate overridden to {args.lr}")
    if args.lora_r is not None:
        _grpo.LORA_KWARGS["r"] = args.lora_r
        print(f"[prestage] LoRA rank overridden to {args.lora_r}")
    if args.lora_alpha is not None:
        _grpo.LORA_KWARGS["lora_alpha"] = args.lora_alpha
        print(f"[prestage] LoRA alpha overridden to {args.lora_alpha}")

    started = time.time()
    if args.all or args.model:
        prestage_model()
    if args.all or args.dqn:
        prestage_dqn()
    if args.all or args.act1:
        run_act("act1" + args.suffix, beta=0.0, time_budget_seconds=args.act1_seconds)
    if args.all or args.act4:
        run_act("act4" + args.suffix, beta=0.04, time_budget_seconds=args.act4_seconds)

    print(f"\n[prestage] total {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
