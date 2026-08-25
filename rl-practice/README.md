# Reinforcement learning practice session

CERN STEAM Academy. Two Jupyter notebooks, a support package, and the tooling needed to run
them in front of a room without depending on the network, the hardware, or the day going
well.

Everything below assumes you are in the repository root. Paths in the notebooks are
relative to it, and so is the model snapshot the loader looks for.

---

## 1. What the session is

Audience: strong students, comfortable with supervised deep learning, no RL background
assumed. Two notebooks, each in a solutions version and a student version.

**`notebooks/01_classics_solutions.ipynb` - the classics, 30 minutes.** A tabular
gridworld with known dynamics, where policy evaluation and value iteration are written by
hand, because dynamic programming is the only part of RL where the whole answer is
visible. Then DQN, provided complete, as a *diagnosis* exercise rather than an
implementation one: students run the working configuration and three configurations that
each differ from it by exactly one field, plot the four reward curves on one axis, and
identify which leg of the deadly triad each broken configuration attacks. CPU only; one
configuration takes about 90 seconds and the notebook runs four of them.

**`notebooks/02_grpo_solutions.ipynb` - GRPO in five acts, 60 minutes.** One model
throughout, `Qwen/Qwen2.5-0.5B-Instruct`, trained with GRPO on integer multiplication with
a programmatic reward. Act 0 measures the base model and picks a task difficulty where its
pass rate leaves room for learning. Act 1 trains it hard with the KL penalty switched off.
Act 2 re-runs the same frozen evaluations and shows what that cost in general ability. Act
3 diagnoses the logs. Act 4 changes one line, `beta = 0.04`, and shows the scissors
closing. Act 5 is discussion. This notebook needs the GPU.

The student notebooks are generated from the solutions notebooks by
`tools/make_student.py` (section 5). The solutions notebooks keep their outputs, so the
expected results can be read without rerunning anything.

---

## 2. Quick start

Target environment: `registry.cern.ch/ngt/pytorch:2.3.1` - Python 3.11, torch 2.3.1+cu121,
numpy pinned to 1.24.4 by `PIP_CONSTRAINT=/etc/pip-constraints.txt` inside the image - on a
single NVIDIA T4, compute capability (7, 5), 15.6 GB.

```bash
pip install -c constraints.txt -r requirements.txt   # never omit -c, see section 3
python tools/check_env.py                            # the gate
pytest tests/                                        # includes a real GPU run
```

`tools/check_env.py` prints the versions of everything that matters and then asserts, in
order of how badly a failure ruins your day: torch is still 2.3.x, CUDA is actually
available, the device is compute capability (7, 5), and neither `flash_attn` nor `vllm` is
installed. It finishes with a real fp32 and fp16 matmul on the device, because version
strings can lie and kernels cannot. It exits non-zero on any failure. On non-Turing
development hardware, pass `--allow-any-gpu` to relax the capability assertion only.

`pytest tests/` runs the fast unit tests for the task data and the reward functions, plus
`tests/test_smoke_gpu.py`, which is a real 3-step GRPO run that asserts a finite loss. That
one takes about 60 seconds including loading the model, and skips cleanly when no CUDA
device is present. It is the test that catches an fp16 NaN, an out-of-memory in the log-
probability forward pass, or a TRL release whose internals moved under our subclass -
before a room full of people finds them for you.

A note on precision, since it looks like a bug the first time you check. Turing has no
bf16 tensor cores, but `torch.cuda.is_bf16_supported()` returns `True` on a T4 anyway,
because it falls back to checking that a bfloat16 tensor can be allocated - which it can,
emulated. Trust the compute capability, not the flag. Every configuration in this
repository sets `fp16=True, bf16=False`, and `check_env.py` prints a note when it sees the
combination.

---

## 3. Why `constraints.txt` exists

This is the most important operational point in the document. TRL's `GRPOTrainer` is recent
and pulls modern `transformers` and `accelerate`; the base image ships torch 2.3.1 built
for CUDA 12.1 with sm75 kernels. A plain `pip install -r requirements.txt` gives pip
permission to satisfy some transitive requirement by replacing that torch, and it will take
it. The replacement may be the CPU wheel, in which case `torch.cuda.is_available()` quietly
becomes `False`; or a cu12x build compiled without sm75 kernels, in which case nothing runs
on a T4; or a version whose ABI does not match the CUDA runtime in the image; or it may
simply drag numpy past the 2.0 ABI break, against which torch 2.3.1 segfaults. All of these
fail late, in the notebook, in a way that looks like a bug in the notebook.
`constraints.txt` pins torch, torchvision, torchaudio and numpy so that pip **fails at
install time** instead, naming the package that wanted the upgrade. It exists to convert a
confusing runtime failure into a boring resolver error. Always install with `-c
constraints.txt`, and run `tools/check_env.py` afterwards to confirm nothing moved.

The dependency set in `requirements.txt` was resolved and tested on the target hardware: it
installs without moving torch, torchvision, torchaudio or numpy. Every version is `==`
pinned, including the transitive pins for `huggingface_hub` and `tokenizers`, which are the
two that break `from_pretrained` when they drift. vLLM, flash-attention and bitsandbytes
are deliberately absent - the first would force the torch upgrade this whole file exists to
prevent, the second has no sm75 kernels, and the third is unnecessary for a 0.5B model in
fp32.

---

## 4. Pre-staging

Assume nothing about student network access or patience. `tools/prestage.py` produces every
artefact the notebooks can fall back on, so that nothing has to succeed live.

```bash
python tools/prestage.py --all           # model + dqn + both GRPO reference runs
python tools/prestage.py --model         # just the base weights, roughly 1 GB
python tools/prestage.py --dqn           # just notebook 1, CPU, four runs
python tools/prestage.py --act1 --act4   # just the GRPO reference runs
```

Everything lands under `assets/`:

| Path | What it is |
|---|---|
| `base_model/` | `Qwen/Qwen2.5-0.5B-Instruct`, downloaded once |
| `reference_adapters/act1/` | LoRA adapter from the author's `beta = 0.0` run |
| `reference_adapters/act4/` | LoRA adapter from the author's `beta = 0.04` run |
| `reference_logs/act{1,4}.csv` | per-step metrics from those runs |
| `reference_logs/act{1,4}_summary.json` | steps, wall time, peak memory, before/after numbers |
| `reference_logs/act{1,4}_completions.json` | completions sampled during training, for the Act 3 drift table |
| `snapshots/*.json` | frozen evaluations: `before`, `after_act1`, `after_act4` |
| `dqn/*.pt` | the four notebook-1 runs, weights plus reward curves |

The geometry of these runs - task difficulty, split seeds, time budgets, evaluation
cadence - lives in `rlpractice/grpo.py` and is imported by both the script and the
notebooks, so the numbers printed in the session and the numbers used to build the
reference artefacts cannot drift apart. `--act1-seconds`, `--act4-seconds` and `--lr`
override the defaults for a re-run.

The model snapshot is not an optimisation, it is the offline path.
`rlpractice.grpo.load_model_and_tokenizer` checks for `assets/base_model/config.json`, and
when it is there it passes `local_files_only=True` to `from_pretrained`; only when it is
absent does it fall back to the hub. That path is relative, so it resolves against the
kernel's working directory. Set `RLPRACTICE_MODEL_DIR` to an absolute path when the weights
live on a shared read-only volume, or when Jupyter is started somewhere other than the
repository root.

The reference runs are what makes `TRAIN_FROM_SCRATCH` work:

```python
TRAIN_FROM_SCRATCH = True   # set False to load the pre-staged adapter and logs
```

Every training cell has this flag at the top. Set it to `False` and the cell loads the
reference adapter, log CSV and snapshots instead of training, and every downstream plot,
evaluation and discussion cell still works. That is the insurance policy: a student whose
GPU misbehaves, whose run diverges, or who simply arrived late flips one flag and continues
with the analysis rather than losing the exercise. It is also what makes the notebooks
readable on a laptop. Rehearse the `False` path before the session, because it is the path
you will need under pressure.

Measured costs of pre-staging, for planning: one DQN configuration is about 90 seconds on
CPU and there are four of them; GRPO costs about 4.5 to 5 seconds per optimiser step on the
T4, with a measured peak of 7.09 GB allocated and 8.67 GB reserved. The step count is not
hard-coded anywhere - `TimeBudgetCallback` times the first few steps and derives a budget,
because a hard-coded step count is a promise about hardware you do not control. Act 1 is
given 900 seconds of training and Act 4 is given 720.

---

## 5. Building the student notebooks

```bash
python tools/make_student.py                            # every notebooks/*_solutions.ipynb
python tools/make_student.py SOURCE.ipynb DEST.ipynb     # one specific pair
python tools/make_student.py --strip-package OUT_DIR     # also hole out a copy of rlpractice/
```

In a code cell, a hint comment and a marked block:

```python
# TODO(hint): compute the group-normalised advantage
# BEGIN SOLUTION
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
# END SOLUTION
```

becomes `# TODO: compute the group-normalised advantage` followed by `raise
NotImplementedError`, indented to match the marker so that a hole inside a function body
stays inside the function body. In a markdown cell, `<!-- TODO(hint): ... -->` around a
`<!-- BEGIN SOLUTION -->` block becomes a blockquote inviting an answer. Everything else -
headers, LaTeX, narrative prose - is preserved byte for byte, so the two versions read as
the same document and the solutions can be published afterwards without anyone having to
re-find their place. Outputs are stripped from the student notebooks and kept in the
solutions. The tool is idempotent, and it fails loudly with exit code 2 on an unbalanced or
nested marker rather than silently truncating an exercise.

**Student notebooks are build artefacts. Never hand-edit one.** The next build discards the
edit without saying so, and the two versions drift apart in a way nobody notices until a
room full of people is looking at the wrong cell. Edit the solutions notebook and rebuild.
Get the solutions notebook working end to end first, then generate the student version -
not the other way round.

Notebook *content* is authored as Python rather than JSON: `tools/nbbuild.py` provides
`md()`, `code()`, `write()` and `validate()`, which turn a list of cells into a valid
`.ipynb`. Editing raw notebook JSON is how trailing-newline and execution-count corruption
gets in.

---

## 6. Repository layout

```
rl-practice/
  README.md                    this document
  requirements.txt             the tested dependency set, == pinned throughout
  constraints.txt              the torch/numpy pins that make a bad resolve fail early
  rlpractice/
    __init__.py                nothing here reaches the network at import time
    mdp.py                     gridworld, policy evaluation, value iteration, rendering
    dqn.py                     complete DQN, the four configs students diagnose, and the
                               matplotlib CartPole viewer for watching an episode
    arithmetic.py              the multiplication task, generated from a seed
    rewards.py                 format and correctness rewards - the entire objective
    evaluation.py              FROZEN before/after measurements; do not parameterise
    general_text.py            embedded prose corpus the forgetting probe reads
    grpo.py                    model loading and the shared GRPO geometry
    callbacks.py               instrumented trainer, metrics CSV, NaN guard, time budget
    dashboard.py               the six-panel training dashboard and the scissors plot
  notebooks/
    01_classics_solutions.ipynb   authored here, outputs kept
    01_classics_student.ipynb     build artefact, do not edit
    02_grpo_solutions.ipynb       authored here, outputs kept
    02_grpo_student.ipynb         build artefact, do not edit
  tools/
    check_env.py               the environment gate; run it first, in a fresh container
    prestage.py                downloads the model, runs the reference training
    make_student.py            strips solution markers into the student notebooks
    nbbuild.py                 builds .ipynb files from Python cell lists
  tests/
    test_arithmetic.py         data generation, answer parsing, train/eval leakage
    test_rewards.py            the reward battery students run against their own code
    test_smoke_gpu.py          a real 3-step GRPO run; skipped without CUDA
  assets/                      produced by tools/prestage.py; see section 4
```

---

## 7. Troubleshooting

**`torch.cuda.is_available()` is False.** Check the torch version first:
`python -c "import torch; print(torch.__version__, torch.version.cuda)"`. If it does not
start with `2.3.1`, pip replaced it - reinstall with `-c constraints.txt` and read section
3. If the version is correct, the GPU is not visible to the container: start it with
`--gpus all` and confirm with `nvidia-smi` from *inside* it. Notebook 1 runs on CPU
regardless; notebook 2 does not.

**CUDA out of memory.** The measured peak for the GRPO runs is 7.09 GB allocated and 8.67
GB reserved on a 15.6 GB card, so a fresh device has ample headroom and an OOM usually
means something else is holding the card - most often an earlier notebook kernel that still
has a model resident. `nvidia-smi` names the process; restarting that kernel is the fix. If
you have genuinely changed the geometry, cut `max_completion_length` in
`rlpractice.grpo.GRPO_COMMON` before touching anything else: the log-probability forward
pass materialises a `batch x seq_len x 152k` logits tensor, and completion length is the
term in it you control most cheaply. Do not raise `per_device_train_batch_size` to use up
apparently free memory - it is the forward-pass batch, not the generation batch, and it
must stay a multiple of `num_generations`.

**NaN loss.** `NaNGuardCallback` raises immediately and names the step. An occasional
`grad_norm: nan` in the logs is not a failure: fp16 training runs a `GradScaler` whose job
is to overshoot, notice the overflow, skip the step and back off, so the guard only fires
after eight consecutive bad steps, or on a non-finite loss, or on a corrupted parameter. If
it fires early and repeatedly, check that the model was loaded in fp32.
`grpo.load_model_and_tokenizer` does this deliberately: fp32 master weights with
`fp16=True` in the training config means autocast plus loss scaling, whereas fp16 weights
*and* `fp16=True` is the classic NaN generator. Confirm `bf16=False` too - see the Turing
note in section 2.

**Model download blocked.** Pre-stage `assets/base_model/` on a machine that does have
network access and copy the directory across; the loader switches to `local_files_only=True`
as soon as `assets/base_model/config.json` exists. For a shared read-only volume, point
`RLPRACTICE_MODEL_DIR` at it with an absolute path. Note that `TRAIN_FROM_SCRATCH = False`
does not rescue you here: it skips the training, not the model, since the frozen evaluations
still need the base weights. The weights are the one thing that must be present.

**The notebook kernel cannot `import rlpractice`.** There is no install step for the
package; it is imported from the repository root, which must therefore be on `sys.path`.
Start Jupyter from the repository root, or set
`PYTHONPATH=/path/to/rl-practice`. The same mistake shows up a second way: if the kernel's
working directory is `notebooks/`, the relative path `assets/base_model` does not resolve
either, and the model loader silently falls back to the hub instead of the local snapshot.
Put `import os; print(os.getcwd())` in the first cell when either symptom appears.
