"""Instrumentation for GRPO training: metrics logging, a NaN guard, a time budget.

Three things live here.

``InstrumentedGRPOTrainer``
    A thin subclass of TRL's ``GRPOTrainer`` that records two quantities TRL does not
    log: policy entropy and the fraction of groups whose rewards have zero standard
    deviation. Both are central to Act 3, and neither can be recovered from outside the
    trainer, so a subclass it is. It touches nothing else.

``MetricsCSVCallback``
    Appends one row per step to a CSV, and periodically runs the *frozen* evaluation
    functions so that the scissors plot has both blades. Writing to CSV rather than
    holding a list in memory means a crashed kernel does not cost you the run.

``NaNGuardCallback``
    Raises immediately, naming the step, if the loss stops being finite. fp16 GRPO on
    Turing can produce NaNs; a silent NaN halfway through a live session is much worse
    than a loud failure, because everything downstream keeps running and quietly plots
    garbage.

``TimeBudgetCallback``
    Measures the real cost of the first few steps and then stops training when the
    measured budget is spent. The step count is measured, not guessed, because a
    hard-coded step count is a promise about hardware you do not control.
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any

import torch
from transformers import TrainerCallback

from . import dashboard, evaluation

try:  # TRL is only needed for the trainer subclass; the callbacks work without it.
    from trl import GRPOTrainer
except Exception:  # pragma: no cover - exercised only in a TRL-less environment
    GRPOTrainer = object  # type: ignore[assignment,misc]


class InstrumentedGRPOTrainer(GRPOTrainer):  # type: ignore[misc,valid-type]
    """GRPOTrainer that additionally logs policy entropy and zero-advantage groups.

    **Entropy.** ``_get_per_token_logps`` returns the log-probability the policy assigns
    to each token it actually sampled, so the mean of ``-log p(sampled token)`` is a
    Monte-Carlo estimate of the policy's per-token entropy:

    .. math:: H(\\pi) = -\\sum_x \\pi(x)\\log \\pi(x) = \\mathbb{E}_{x\\sim\\pi}[-\\log \\pi(x)]

    It is worth computing this way rather than from the full logits tensor, which for a
    152k vocabulary is precisely the thing we cannot afford to materialise twice.

    It is a **proxy, not an unbiased estimate**, and the reason is worth knowing because
    it is a real trap in TRL 0.15.2. The identity above needs the samples to come from
    :math:`\\pi` itself. They do not: TRL constructs its ``GenerationConfig`` without
    setting ``top_k``, so it inherits the transformers class default of 50 and GRPO
    samples from the top-50 renormalised distribution :math:`q`. Measured on this base
    model, :math:`\\mathbb{E}_q[-\\log \\pi]` reads about 1.73 nats against a true
    :math:`H(\\pi)` of about 2.24 - roughly 77% of it - because the truncated tail is
    exactly the high-surprisal part. The bias shrinks as the policy sharpens and the
    top-50 mass approaches 1, so the measured entropy *fall* is compressed relative to
    the true one. The quantity is still monotone in what Act 3 asks about, which is all
    that is needed there; it is not the number to quote in a paper.

    Two consequences worth noticing rather than papering over: the sampling temperature
    must stay at 1.0 for even this proxy to mean anything, and GRPO's importance ratio
    formally assumes samples drawn from the policy - top-k truncation quietly violates
    that assumption in every TRL GRPO run at this version.

    **Zero-std groups.** TRL computes ``advantages = (r - mean_group) / (std_group + eps)``.
    A group whose rewards are all identical has zero numerator, hence exactly zero
    advantage for every member, hence contributes no policy gradient at all. So the
    fraction of groups with zero reward std equals the fraction of groups whose
    advantages are all zero, which we can read straight off the prepared inputs without
    reaching into TRL's reward computation. When this fraction is high, the effective
    batch size is far smaller than it looks, and that is the single most common reason a
    GRPO run "trains" for an hour and learns nothing.
    """

    #: Record a few raw completions every this many steps. Act 3 prints them as a table so
    #: that the drift in what the model actually writes is legible, not merely plotted.
    #: There is no other way to get at them: TRL decodes completions inside
    #: ``_prepare_inputs`` and does not keep them.
    completion_log_every: int = 25
    #: How many completions to keep from each sampled step.
    completion_log_n: int = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialised here, not lazily in _prepare_inputs, so that completion_table()
        # works on a trainer that was constructed to hold a pre-staged adapter and never
        # had train() called on it - which is exactly the TRAIN_FROM_SCRATCH = False path.
        self.completion_samples: dict[int, list[str]] = {}

    def _prepare_inputs(self, inputs):
        prepared = super()._prepare_inputs(inputs)

        advantages = prepared.get("advantages")
        if advantages is not None and advantages.numel() > 0:
            groups = advantages.view(-1, self.num_generations)
            zero_groups = (groups.abs().sum(dim=1) == 0).float().mean().item()
            self._metrics["frac_zero_std_groups"].append(zero_groups)

        step = self.state.global_step
        every = self.completion_log_every
        # `_prepare_inputs` runs once per micro-batch, so twice per optimiser step under
        # gradient accumulation. Keep the first sighting of each step and ignore the rest.
        if every and step % every == 0 and step not in self.completion_samples:
            completion_ids = prepared.get("completion_ids")
            if completion_ids is not None:
                texts = self.processing_class.batch_decode(
                    completion_ids[: self.completion_log_n], skip_special_tokens=True
                )
                self.completion_samples[step] = texts

        return prepared

    def completion_table(self, width: int = 100) -> str:
        """The recorded completions as a printable table, oldest first."""
        lines = []
        for step in sorted(self.completion_samples):
            lines.append(f"--- step {step} " + "-" * max(0, width - 14 - len(str(step))))
            for text in self.completion_samples[step]:
                flat = " ".join(text.split())
                lines.append(f"  {flat[:width]}" + (" ..." if len(flat) > width else ""))
        return "\n".join(lines)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # The flag tells _get_per_token_logps that the pass it is about to do is the
        # policy pass, not the reference pass (which happens inside _prepare_inputs).
        self._recording_entropy = True
        try:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            self._recording_entropy = False

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        logps = super()._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
        )
        if getattr(self, "_recording_entropy", False):
            # attention_mask covers prompt + completion; the last `logits_to_keep`
            # columns are exactly the completion mask TRL built.
            mask = attention_mask[:, -logits_to_keep:].to(logps.dtype)
            n_tokens = mask.sum()
            if n_tokens > 0:
                entropy = -(logps.detach() * mask).sum() / n_tokens
                self._metrics["entropy"].append(entropy.item())
        return logps


class NaNGuardCallback(TrainerCallback):
    """Fail loudly and immediately when training has gone numerically wrong.

    A word on why this watches three things rather than one.

    **The GRPO loss is not a useful health signal.** With one inner iteration,
    ``exp(logp - logp.detach())`` is identically 1, so the per-token loss is just
    ``-advantage``, and advantages are mean-centred within each group by construction.
    The reported loss is therefore approximately zero on *every* step of a healthy run,
    and it would also be approximately zero on a run that had stopped learning entirely.
    Watching only the loss would produce a guard that essentially never fires. It is
    still checked, because a NaN there is unambiguous.

    **A non-finite gradient norm is normal, in moderation.** fp16 training runs a
    ``GradScaler``, whose entire job is to push gradients up into fp16's representable
    range, notice the overflows, skip those steps, and back the scale off. Seeing
    ``grad_norm: nan`` occasionally means the scaler is working. Seeing it on many
    consecutive steps means it has stopped being able to find a workable scale, and
    training has silently become a very expensive no-op.

    **Corrupt weights are the thing that actually ruins the session.** Once a trainable
    parameter contains NaN, every number downstream - rewards, perplexity, the scissors
    plot - is meaningless, and nothing else will tell you.
    """

    def __init__(self, check_parameters: bool = True, max_consecutive_bad_grads: int = 8):
        self.check_parameters = check_parameters
        self.max_consecutive_bad_grads = max_consecutive_bad_grads
        self.consecutive_bad_grads = 0

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        logs = logs or {}

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            if math.isfinite(float(grad_norm)):
                self.consecutive_bad_grads = 0
            else:
                self.consecutive_bad_grads += 1
                if self.consecutive_bad_grads >= self.max_consecutive_bad_grads:
                    raise FloatingPointError(
                        f"The gradient norm has been non-finite for "
                        f"{self.consecutive_bad_grads} consecutive steps, ending at step "
                        f"{state.global_step}. The fp16 GradScaler skips overflowing steps, "
                        "so training has not actually been updating the model. Lower the "
                        "learning rate, or check that the model was loaded in fp32."
                    )

        loss = logs.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            raise FloatingPointError(
                f"Loss became non-finite ({loss}) at step {state.global_step}.\n"
                "This is the classic fp16 failure. Check, in order:\n"
                "  1. the model was loaded in fp32 (torch_dtype=torch.float32) while the\n"
                "     training config sets fp16=True - loading fp16 weights AND enabling\n"
                "     fp16 is the usual cause;\n"
                "  2. bf16 is off (Turing sm75 has no bf16 support);\n"
                "  3. the learning rate is not absurd for this model size."
            )
        if self.check_parameters and model is not None:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and not torch.isfinite(parameter.data).all():
                    raise FloatingPointError(
                        f"Trainable parameter '{name}' contains non-finite values at "
                        f"step {state.global_step}. The weights are already corrupted; "
                        "restart from the base model."
                    )


class TimeBudgetCallback(TrainerCallback):
    """Convert a wall-clock budget into a step budget, by measurement.

    A hard-coded ``max_steps`` is a promise about hardware. This measures the cost of the
    first ``calibration_steps`` real steps, projects how many fit in the budget, prints
    the projection so nobody is surprised, and stops training there.

    ``TrainingArguments.max_steps`` should be set to ``max_steps_cap`` so that the
    learning-rate schedule is well defined; this callback stops earlier by setting
    ``control.should_training_stop``. Use a constant (or constant-with-warmup) schedule
    so that stopping early does not leave the learning rate somewhere arbitrary - that
    would break the single-variable comparison between Act 1 and Act 4.
    """

    def __init__(
        self,
        time_budget_seconds: float,
        max_steps_cap: int,
        calibration_steps: int = 3,
        warmup_steps: int = 1,
        wallclock_backstop: float = 1.4,
        verbose: bool = True,
    ):
        self.time_budget_seconds = float(time_budget_seconds)
        self.max_steps_cap = int(max_steps_cap)
        self.calibration_steps = int(calibration_steps)
        self.warmup_steps = int(warmup_steps)
        self.wallclock_backstop = float(wallclock_backstop)
        self.verbose = verbose

        self.t_start: float | None = None
        self.t_calibration_start: float | None = None
        self.calibration_start_step: int = 0
        self.seconds_per_step: float | None = None
        self.budget_steps: int | None = None
        self.stopped_because: str | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_start = time.time()
        self.t_calibration_start = None
        if self.verbose:
            print(
                f"[time budget] {self.time_budget_seconds:.0f} s, cap {self.max_steps_cap} steps. "
                f"Calibrating on steps {self.warmup_steps + 1}-"
                f"{self.warmup_steps + self.calibration_steps}..."
            )

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        now = time.time()

        # The first step is not representative and must not be measured. It carries
        # CUDA context creation, cuBLAS and SDPA kernel autotuning, the dataloader
        # spinning up its workers, and the first (slow) allocations into an empty
        # caching allocator. Measured against a steady-state cost of about 5 s/step it
        # came out at 17 s/step, which projected a budget three times too small and cut
        # a 15-minute run down to 4.6 minutes. Discard the warm-up, then measure.
        if self.t_calibration_start is None and step >= self.warmup_steps:
            self.t_calibration_start = now
            self.calibration_start_step = step
            return control

        if self.budget_steps is None and step >= self.warmup_steps + self.calibration_steps:
            elapsed = now - self.t_calibration_start
            # Divide by the number of steps the window actually spans. With
            # warmup_steps = 0 the window opens after step 1 has already completed, so it
            # covers one step fewer than `calibration_steps`.
            measured_steps = step - self.calibration_start_step
            self.seconds_per_step = elapsed / max(1, measured_steps)
            projected = int(self.time_budget_seconds / self.seconds_per_step)
            self.budget_steps = max(1, min(self.max_steps_cap, projected))
            if self.verbose:
                print(
                    f"[time budget] measured {self.seconds_per_step:.2f} s/step over "
                    f"{self.calibration_steps} steps (warm-up discarded) -> budget is "
                    f"min({self.max_steps_cap}, {projected}) = {self.budget_steps} steps "
                    f"(~{self.budget_steps * self.seconds_per_step / 60:.1f} min of training)."
                )

        # The wall-clock budget is authoritative; the projection above is advisory.
        #
        # This is not the obvious design and it was arrived at by getting it wrong twice.
        # Projecting a step count from the first few steps assumes the cost per step is
        # stationary, and here it emphatically is not: the *untrained* model rambles for
        # 50-70 completion tokens while the trained one emits about 18, and generation
        # time scales with that. Measured, the first steps cost around 14 s while the
        # steady state is nearer 5 s, so the projection came out three times pessimistic
        # and cut a fifteen-minute budget down to five minutes of actual training.
        #
        # Enforcing the wall clock directly gives the thing we actually promised - "this
        # cell takes fifteen minutes" - and is robust to step cost changing in either
        # direction during the run. The step cap still bounds the run on fast hardware.
        elapsed = now - self.t_start
        if step >= self.max_steps_cap:
            self.stopped_because = f"step cap reached ({self.max_steps_cap} steps)"
            control.should_training_stop = True
        elif elapsed >= self.time_budget_seconds:
            self.stopped_because = (
                f"time budget spent ({elapsed:.0f} s over {step} steps, "
                f"{elapsed / max(1, step):.1f} s/step actual)"
            )
            control.should_training_stop = True

        if control.should_training_stop and self.verbose:
            print(f"[time budget] stopping at step {step}: {self.stopped_because}")
        return control


class MetricsCSVCallback(TrainerCallback):
    """One CSV row per step, plus periodic frozen evaluation.

    The expensive part is the evaluation, so the two probes run at different cadences:

    * ``ppl_every`` - general-text perplexity. One forward pass over ~3k tokens. Cheap,
      so run it often; it is the blade of the scissors that moves fastest.
    * ``acc_every`` - held-out task accuracy. Greedy generation over ``acc_n`` problems.
      Expensive, so run it rarely and on a fixed prefix of the eval set, which keeps it
      comparable to the full endpoint evaluations.

    Both call the frozen functions in ``evaluation.py`` unchanged. That is the whole
    point of freezing them.
    """

    def __init__(
        self,
        csv_path: str | os.PathLike,
        tokenizer,
        eval_dataset,
        ppl_every: int = 5,
        acc_every: int = 20,
        acc_n: int = 32,
        dashboard_every: int | None = 10,
        baseline_ppl: float | None = None,
        baseline_accuracy: float | None = None,
        title: str = "GRPO training",
        verbose: bool = True,
    ):
        self.csv_path = str(csv_path)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.ppl_every = ppl_every
        self.acc_every = acc_every
        self.acc_n = acc_n
        self.dashboard_every = dashboard_every
        self.baseline_ppl = baseline_ppl
        self.baseline_accuracy = baseline_accuracy
        self.title = title
        self.verbose = verbose

        self.t_start: float | None = None
        self.eval_seconds = 0.0
        self.rows: list[dict[str, Any]] = []

    # -- csv plumbing --------------------------------------------------------------

    def _write_header(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=dashboard.LOG_COLUMNS).writeheader()

    def _append(self, row: dict[str, Any]):
        full = {column: row.get(column, "") for column in dashboard.LOG_COLUMNS}
        with open(self.csv_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=dashboard.LOG_COLUMNS).writerow(full)
        self.rows.append(full)

    # -- hooks ---------------------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.t_start = time.time()
        self._write_header()

        # Step 0 is the pre-training baseline, and it must be measured with the *same*
        # estimator as every later point on the curve. The snapshot baseline is taken at
        # n = 64 while the in-training probe uses acc_n = 32; plotting one against the
        # other put a 0.06 step into a curve whose whole claim is a 0.22 rise, purely
        # because the estimator changed. So measure step 0 here, at acc_n. The n = 64
        # snapshot value is still drawn, as the dotted reference line.
        model = model if model is not None else kwargs.get("model")
        baseline_point = ""
        if model is not None and self.acc_every:
            baseline_point = evaluation.eval_task_accuracy(
                model, self.tokenizer, self.eval_dataset, n=self.acc_n
            )
        elif self.baseline_accuracy is not None:
            baseline_point = self.baseline_accuracy

        self._append(
            {
                "step": 0,
                "wall_seconds": 0.0,
                "general_ppl": self.baseline_ppl if self.baseline_ppl is not None else "",
                "task_accuracy": baseline_point,
            }
        )

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:  # the final summary log at the end of training
            return
        step = state.global_step
        model = model if model is not None else kwargs.get("model")

        row: dict[str, Any] = {
            "step": step,
            "wall_seconds": round(time.time() - self.t_start - self.eval_seconds, 2),
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "reward_mean": logs.get("reward"),
            "reward_std": logs.get("reward_std"),
            "frac_zero_std_groups": logs.get("frac_zero_std_groups"),
            "completion_length_mean": logs.get("completion_length"),
            "entropy": logs.get("entropy"),
            "kl": logs.get("kl"),
        }

        if model is not None and self.ppl_every and step % self.ppl_every == 0:
            t0 = time.time()
            row["general_ppl"] = evaluation.eval_general_perplexity(model, self.tokenizer)
            self.eval_seconds += time.time() - t0

        if model is not None and self.acc_every and step % self.acc_every == 0:
            t0 = time.time()
            row["task_accuracy"] = evaluation.eval_task_accuracy(
                model, self.tokenizer, self.eval_dataset, n=self.acc_n
            )
            self.eval_seconds += time.time() - t0

        self._append(row)

        if self.dashboard_every and step % self.dashboard_every == 0:
            try:
                dashboard.redraw(
                    self.csv_path,
                    title=self.title,
                    baseline_ppl=self.baseline_ppl,
                    baseline_accuracy=self.baseline_accuracy,
                )
            except Exception as error:  # a broken plot must never kill a training run
                print(f"[dashboard] redraw failed, continuing: {error}")
        elif self.verbose:
            parts = [f"step {step:4d}"]
            for key, label, fmt in (
                ("reward_mean", "reward", "{:.3f}"),
                ("kl", "kl", "{:.4f}"),
                ("entropy", "H", "{:.3f}"),
                ("completion_length_mean", "len", "{:.0f}"),
                ("frac_zero_std_groups", "zero-std", "{:.2f}"),
                ("general_ppl", "ppl", "{:.2f}"),
                ("task_accuracy", "acc", "{:.3f}"),
            ):
                value = row.get(key)
                if value not in (None, ""):
                    parts.append(f"{label}={fmt.format(float(value))}")
            print("  ".join(parts))

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.verbose and self.t_start is not None:
            total = time.time() - self.t_start
            print(
                f"[metrics] {len(self.rows)} rows -> {self.csv_path}; "
                f"{total:.0f} s total, {self.eval_seconds:.0f} s of that in evaluation"
            )


def peak_memory_report(reset: bool = False) -> dict[str, float]:
    """Peak CUDA memory in GB, as a dict. Print this after every training act.

    The T4 has 16 GB and the acceptance criterion is under 14. Reporting it is not
    optional: the vocabulary-sized logits tensor is the whole reason the batch geometry
    in the notebooks looks the way it does, and students should see the receipt.
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": float("nan"), "reserved_gb": float("nan")}
    report = {
        "allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return report
