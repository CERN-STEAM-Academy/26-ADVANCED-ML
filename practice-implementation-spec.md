# Implementation spec: RL lecture practice session

Target: two Jupyter notebooks, solutions plus student versions, running end to end on a single
NVIDIA T4 inside `registry.cern.ch/ngt/pytorch:2.3.1`.

Audience: strong students, comfortable with supervised deep learning, no RL background assumed.
Duration: 35 min for notebook 1, 85 min for notebook 2.

---

## 0. Read this before writing any code

Three constraints determine the whole design. Resolve them first, in this order, and stop to
report if any of them fails.

### 0.1 Dependency resolution is task one

The image ships **PyTorch 2.3.1**. TRL's `GRPOTrainer` is recent and pulls modern
`transformers` / `accelerate`. The failure mode to avoid: pip silently upgrades torch, the
CUDA build changes, and nothing works on sm75 any more.

Procedure:

1. In a fresh container, `pip install` a candidate set with pinned versions.
2. **Immediately assert** `torch.__version__` still starts with `2.3.1` and
   `torch.cuda.is_available()` is True. If torch moved, pin harder or use `--no-deps` and
   install the transitive deps by hand.
3. Record the exact working set in `requirements.txt` with `==` pins for everything.
4. Print `torch.cuda.get_device_capability()` and assert it is `(7, 5)`.

Starting point for the candidate set, **unverified, expect to iterate**:

```
transformers==4.49.0
trl==0.15.2
peft==0.14.0
accelerate==1.4.0
datasets==3.3.2
```

Constraints that are not negotiable:
- **No vLLM.** It will force a torch upgrade. Set `use_vllm=False`.
- **No flash-attention.** sm75 is unsupported. Use `attn_implementation="sdpa"`.
- **No bf16 anywhere.** Turing has no bf16. `bf16=False`, `fp16=True`.

If a clean pinned set inside the base image proves impossible after honest effort, say so
and stop. A custom image is then justified, and that is a decision for the author, not a
workaround to improvise.

### 0.2 The vocabulary is the memory bottleneck, not the model

Qwen2.5 has a vocabulary of roughly 152k tokens. The logits tensor during the log-prob
forward pass is `batch × seq_len × 152k`. At 64 sequences of 250 tokens in fp32 that is
about 11 GB for the logits alone, and the T4 has 16 GB. This will OOM if configured naively.

The fix is that **generation batch and forward batch are different things**. Generation of 8
completions is cheap; the forward pass over them is not. Therefore:

```python
num_generations              = 8
per_device_train_batch_size  = 8      # must be divisible by num_generations
gradient_accumulation_steps  = 2      # -> 2 prompts per optimiser step
max_prompt_length            = 160
max_completion_length        = 128
```

Verify actual peak memory with `torch.cuda.max_memory_allocated()` and report it. If it
exceeds ~13 GB, cut `max_completion_length` before touching anything else.

### 0.3 fp16 instability is expected, so design against it

Load the model in **fp32** and let `fp16=True` in the training config drive autocast plus
`GradScaler`. Do **not** load fp16 weights and also enable `fp16=True` — that combination is
the classic NaN generator. A 0.5B model in fp32 is about 2 GB of weights, which fits fine.

Add an explicit NaN guard: a callback that checks the loss each step and raises a clear
error naming the step number if it is not finite. A silent NaN halfway through a live
session is much worse than a loud failure.

---

## 1. Repository layout

```
rl-practice/
  requirements.txt
  README.md                      # setup, expected runtimes, pre-staging instructions
  rlpractice/
    __init__.py
    arithmetic.py                # task data generation
    rewards.py                   # reward functions
    evaluation.py               # FROZEN eval functions, see section 4
    dashboard.py                 # metrics logging + plotting
    callbacks.py                 # GRPO metric callback, NaN guard
    general_text.py              # embedded general-prose corpus, no download
  notebooks/
    01_classics_solutions.ipynb
    01_classics_student.ipynb
    02_grpo_solutions.ipynb
    02_grpo_student.ipynb
  assets/
    reference_logs/              # pre-staged CSVs, see section 6
    reference_adapters/          # pre-staged LoRA adapters, see section 6
  tools/
    make_student.py              # strips solutions -> student, see section 5
    prestage.py                  # downloads base model, runs reference training
  tests/
    test_rewards.py
    test_arithmetic.py
    test_smoke_gpu.py            # 3-step GRPO run, asserts finite loss
```

**Build the solutions notebooks first and get them working end to end. Only then generate the
student versions.** Never hand-edit a student notebook; it is a build artefact.

---

## 2. Notebook 1: classics (35 min)

Reuse the author's existing notebook structure and content where it exists: MDP env, policy
evaluation, value iteration, epsilon-greedy, replay buffer, DQN on a Gym environment. Keep
the existing style — markdown section headers, an image or equation per section, function
docstrings with an `...` or `# TODO` hole, and a test cell immediately after each exercise.

Two changes from the original:

**2.1 Keep the dynamic programming warm-up as an implementation exercise.** Policy
evaluation and value iteration are quick, self-contained, and worth writing by hand. 10 min.

**2.2 Replace "implement DQN from scratch" with a diagnosis exercise.** 25 min.

Provide `train_dqn(env, config)` fully implemented. Then provide three configs that differ
from the working one by exactly one thing each:

| Config | Single change | Which leg of the triad |
|---|---|---|
| `CONFIG_A` | `target_update_every = 1` (target follows policy immediately) | bootstrapping target chases itself |
| `CONFIG_B` | `buffer_size = 32` | correlated samples, off-policy replay defeated |
| `CONFIG_C` | `lr = 1e-1` | function approximation diverges |

Students run all four, plot the four reward curves on one axis, and answer in a markdown
cell: which leg of the deadly triad did each config break, and why does that produce the
curve shape they observed?

Use a small fast environment (CartPole is fine) so all four runs finish in a few minutes.
**Ship pre-trained weights** for the working config so nobody waits on a full run.

---

## 3. Notebook 2: GRPO in five acts (85 min)

This is the centrepiece. One model throughout, five acts, and every act re-uses the *same*
frozen eval functions so the before/after comparison is valid.

### The task

Multiply two integers, answer inside tags. Reward is format compliance plus exact match.

Generated, not downloaded. `rlpractice/arithmetic.py`:

```python
def make_dataset(n, digits_a, digits_b, seed) -> datasets.Dataset
```

Each row: `{"prompt": <chat-formatted question>, "answer": <int as str>}`.
The prompt instructs the model to reason inside `<think></think>` and give the final number
inside `<answer></answer>`, following the DeepSeek-R1-Zero template style.

Hold out a fixed eval split with its own seed, never used for training.

### Act 0 — Meet the model (12 min)

1. Load `Qwen/Qwen2.5-0.5B-Instruct`, fp32, sdpa. Print parameter count and memory.
2. Generate on 3 arithmetic prompts and 5 general prompts. Print raw output. Students should
   *see* the base behaviour before anything changes.
3. **Difficulty sweep.** For `(digits_a, digits_b)` in `[(1,1), (2,1), (2,2), (3,2)]`,
   measure base pass@1 on 32 held-out problems each. Plot as a bar chart.
4. **Student exercise**: pick the difficulty whose pass rate is in the 20–50% band, and
   explain in a markdown cell why both a 5% and a 95% pass rate would make GRPO learn
   nothing. This is the zero-advantage lesson arriving as setup rather than as a separate
   experiment.
5. Save all "before" artefacts to disk: task accuracy, general perplexity, and the 5 general
   generations verbatim.

### Act 1 — Train it, hard (22 min)

GRPO with the KL leash off. Config:

```python
learning_rate = 5e-5     # deliberately hot
beta          = 0.0      # no KL penalty
temperature   = 1.0
lora          = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
reward_funcs  = [format_reward, correctness_reward]
```

**Student holes**: implement `format_reward` and `correctness_reward` in `rewards.py`.
Signature must match TRL's expectation (`completions`, `**kwargs` including the ground-truth
column). Provide unit tests in `tests/test_rewards.py` that students can run to check
themselves — a reward function that silently returns the wrong shape is a miserable thing to
debug live.

**Step budget must be measured, not hard-coded.** Time the first 3 steps, then:

```python
max_steps = min(MAX_STEPS_CAP, int(TIME_BUDGET_SECONDS / measured_seconds_per_step))
```

Print the resulting budget so students know what to expect. `TIME_BUDGET_SECONDS` for act 1
is 900 (15 min of the 22).

Live dashboard updates every N steps (section 4.2).

### Act 2 — The reveal (10 min)

Re-run Act 0's eval cells verbatim. Two artefacts:

1. **Side-by-side generations.** The 5 general prompts, before and after, printed adjacently.
   No judge model needed; the degradation should be visible.
2. **The scissors plot.** Task accuracy and general-text perplexity on twin axes against
   training step. This is the figure previewed in the theory slides.

**Student hole**: write the markdown interpretation. What improved, what degraded, and was
anything in the training objective asking for the second thing?

### Act 3 — Diagnose (12 min)

No training. Analysis of the logs from Act 1:

- KL divergence to the reference model over training. It was never penalised, so watch it climb.
- Policy entropy over training. Watch it fall.
- Fraction of groups with `std(rewards) == 0`.
- Mean completion length.
- Sampled completions at steps 0, 25, 50, final — printed as a table so the drift is legible.

**Student hole**, the most important one in the notebook: a markdown cell answering *which
quantity was unbounded, and why did leaving it unbounded permit this*. This is the transfer
lesson; everything else is mechanics a model can generate.

### Act 4 — The fix (20 min)

Same base model, same everything, **one changed line**: `beta = 0.04`.

Keep the learning rate identical to Act 1. A clean single-variable ablation is worth more
than a better-performing run. Budget fewer steps than Act 1 (`TIME_BUDGET_SECONDS = 720`);
the trend is the point, not convergence.

Re-run the frozen evals. Overlay the new scissors plot on Act 1's. The blades should close:
task accuracy still climbs, perplexity holds much flatter.

**Student hole**: implement the comparison plot given both log CSVs.

### Act 5 — Discussion (9 min)

Markdown only, no code. Prompts for discussion, with space for student answers:

- The KL penalty is one fix. Mixing general data into the RL batch is another, and is what
  labs actually do. Why might that work better?
- With LoRA there is a trivial way to avoid forgetting entirely: never merge the adapter.
  What does that buy, and where does it stop working?
- DeepSeek-R1 used a clip ratio of 10, not the 0.2 typical of classical PPO. What does that
  tell you about transferring hyperparameters between domains?
- R1-Zero needed a 32B+ base model before pure RL produced emergent reasoning. Given that,
  what should you *not* conclude from the 0.5B run you just did?

---

## 4. Shared modules

### 4.1 `evaluation.py` — write this first and freeze it

Acts 0, 2 and 4 must call **byte-identical** eval code, or the comparison is worthless. Write
these once, with fixed seeds and greedy decoding, and never parameterise them in a way that
lets a later act drift.

```python
EVAL_SEED = 1234
GENERAL_PROMPTS = [...]   # exactly 5, fixed, module-level constant

def eval_task_accuracy(model, tokenizer, eval_dataset, n=64) -> float
def eval_general_perplexity(model, tokenizer) -> float
def sample_general_generations(model, tokenizer) -> list[str]
def snapshot(model, tokenizer, eval_dataset, tag) -> dict   # calls all three, saves JSON
```

`eval_general_perplexity` runs a forward pass over an embedded prose corpus and returns mean
token NLL exponentiated. **No generation, no download** — one forward pass, so it is cheap
enough to call every N steps during training.

### 4.2 `general_text.py`

An embedded corpus of general English prose, roughly 2000 to 4000 tokens, as a module-level
string constant. Encyclopaedic and narrative prose, deliberately unrelated to arithmetic.
Must be text the author is free to redistribute — write it or use clearly public-domain
material. **No dataset download**, since network access from the student VMs is unverified.

### 4.3 `callbacks.py`

A `TrainerCallback` that appends one row per step to a CSV: step, reward mean, reward std,
fraction of zero-std groups, mean completion length, approximate policy entropy, KL to
reference. Plus general perplexity every `ppl_every` steps.

A `NaNGuardCallback` that raises with a clear message if the loss is not finite.

### 4.4 `dashboard.py`

Re-plots the CSV as a 2×3 grid of matplotlib axes, called from the callback. Not a live
animation — clear the output and redraw. Simple and robust beats clever.

---

## 5. Solutions to student versions

`tools/make_student.py` strips marked blocks. Never hand-edit student notebooks.

Convention inside solutions notebooks:

```python
# BEGIN SOLUTION
    actual_implementation()
# END SOLUTION
```

The script replaces each marked block with the `hint` text from a preceding comment:

```python
# TODO(hint): compute the group-normalised advantage
# BEGIN SOLUTION
...
# END SOLUTION
```

becomes

```python
# TODO: compute the group-normalised advantage
raise NotImplementedError
```

Markdown cells, LaTeX, section structure and all narrative text are **preserved identically**
in both versions. Only code inside solution markers is removed. Cells that must be answered
in prose get a marked empty markdown cell in the student version.

The script must be idempotent and must fail loudly on unbalanced markers.

Also strip outputs from student notebooks, but **keep outputs in the solutions notebooks** so
the author can see expected results without rerunning.

---

## 6. Pre-staging

Assume nothing about student network access or patience.

`tools/prestage.py` produces, into `assets/`:

1. **Base model weights** downloaded once to a known local path. Notebooks load from that
   path with `local_files_only=True` if present, falling back to the hub otherwise. If a
   shared read-only volume is available, point there; otherwise this is a strong argument
   for baking the weights into a custom image.
2. **Reference LoRA adapters** from the author's own Act 1 and Act 4 runs.
3. **Reference log CSVs** from those runs.

Every training cell gets a flag at the top of the notebook:

```python
TRAIN_FROM_SCRATCH = True   # set False to load pre-staged adapter and logs
```

When False, the cell loads the reference artefacts and every downstream plot and eval still
works. This is the insurance policy: if a student's GPU misbehaves or a run diverges, they
flip one flag and continue with the analysis rather than losing the exercise.

---

## 7. End-to-end test protocol

Do not report success until all of the following pass **inside the target image on the T4**.

1. `pip install -r requirements.txt` in a clean container, then assert torch is still 2.3.1
   and CUDA is available on a (7,5) device.
2. `pytest tests/` green, including the 3-step GPU smoke test.
3. `01_classics_solutions.ipynb` executes top to bottom via `jupyter nbconvert --execute`.
   Record wall time.
4. `02_grpo_solutions.ipynb` executes top to bottom the same way, with real step budgets.
   Record wall time and peak GPU memory.
5. `tools/make_student.py` runs, and both student notebooks execute top to bottom with the
   TODO cells raising `NotImplementedError` at the expected places and nowhere else.
6. Student notebooks with `TRAIN_FROM_SCRATCH = False` execute top to bottom **with no
   NotImplementedError**, using pre-staged artefacts, proving the fallback path works.

Report actual measured numbers, not estimates: per-step seconds, total runtime per act, peak
memory, and the final scissors-plot values for both runs.

---

## 8. Acceptance criteria

The session works only if these hold. Measure and report each.

1. **Act 1 shows reward improvement.** Task reward measurably above baseline within the
   15-minute budget.
2. **Act 2 forgetting is obvious.** General perplexity rises by a clear margin, and at least
   3 of the 5 side-by-side generations are visibly worse to a naive reader. If the
   degradation is marginal, escalate: raise LR, widen LoRA target modules, extend steps,
   choose more distant eval prompts. **Report the setting that made it unambiguous.** This is
   the spine of the session; mild is failure.
3. **Act 4 fix is visible.** With `beta=0.04` and everything else identical, perplexity rises
   substantially less while task reward still improves.
4. **Total notebook 2 runtime under 85 minutes** including all evals, on one T4.
5. **Peak memory under 14 GB.**
6. No NaNs across either full run.

If criterion 2 cannot be met without making the run implausibly long, stop and report. The
notebook design changes at that point, and that is the author's call.

---

## 9. Style notes

- Match the author's existing notebook conventions: markdown headers, LaTeX for equations,
  docstringed functions with holes, a test cell after each exercise.
- Every section gets a short LaTeX or prose introduction explaining *why* before *how*.
- Prefer explicit loops and named intermediates over clever one-liners. Students are reading
  this to learn, not admiring the code.
- Print shapes and sample values liberally. Silent tensors teach nothing.
- No emoji.
