# Setup

For a CERN Kubeflow notebook server built from `registry.cern.ch/ngt/pytorch:2.3.1`.

If you are an instructor setting this up for a class, read section 5 first.

---

## 1. What you need

A notebook server from `registry.cern.ch/ngt/pytorch:2.3.1`, which already provides
Python 3.11, torch 2.3.1+cu121, numpy 1.24.4, matplotlib 3.8.4 and JupyterLab 4.2.5.

| | Notebook 1 (classics) | Notebook 2 (GRPO) |
|---|---|---|
| GPU | not needed, CPU only | **required** |
| compute time, end to end | 2 min 15 s | 15 min 54 s |
| session time, with the exercises | 30 min | 60 min |
| peak GPU memory | - | 7.1 GB |
| disk | negligible | ~1 GB for the model, unless it is on a shared volume |

The whole practice fits in 1 hour 45 minutes with about 20 minutes to spare. Only 18 of
those minutes are the machine working; the rest is reading and doing the exercises.

The GPU this was built and measured on is an NVIDIA T4 (compute capability 7.5, 15.6 GB).
Anything with 10 GB or more should be fine; the timings above will differ.

Request the GPU when you create the notebook server. Adding one later means a new server.

---

## 2. Setup, once

Open a terminal in JupyterLab (File, New, Terminal) and run:

```bash
git clone <this-repository>
cd 26-ADVANCED-ML/rl-practice
pip install --user -c constraints.txt -r requirements.txt
```

`--user` because the image's `site-packages` is root-owned and you cannot write to it.
Installs land in `~/.local`, which is on your persistent volume and survives a server
restart.

**Do not drop the `-c constraints.txt`.** It pins torch, torchvision, torchaudio and numpy
to what the image already ships. Without it, one of the packages being installed can decide
it needs a newer torch, and pip will quietly replace the CUDA build the image was tested
with. That failure surfaces much later as `torch.cuda.is_available()` returning False, or
as kernels that do not exist for this GPU, and it is miserable to diagnose. With the
constraints file, the same situation is an install-time resolver error instead.

The image also sets `PIP_CONSTRAINT=/etc/pip-constraints.txt` (which pins `numpy==1.24.4`).
Passing `-c` on the command line does **not** override that - pip applies both files. This
was verified rather than assumed.

### The model weights

The repository does not contain them. They are about a gigabyte, they are not source, and
thirty people downloading the same gigabyte at once is not a plan. Get them one of two
ways:

```bash
# (a) you were given a shared path - nothing to download, see section 4
# (b) download your own copy, needs network access to huggingface.co
python tools/prestage.py --model
```

Note that `TRAIN_FROM_SCRATCH = False` does **not** avoid this. The reference adapters in
`assets/` are LoRA deltas of 17 MB each and are useless without the base weights underneath.

---

## 3. Check it worked

```bash
python tools/check_env.py
```

It should end with `ENVIRONMENT CHECK PASSED`. It asserts that torch is still 2.3.1, that
CUDA is available, that the device is what you think it is, and that neither vLLM nor
flash-attention got installed (neither works here: flash-attention has no kernels for
Turing, and vLLM would force a torch upgrade).

If you are on a GPU that is not a T4, add `--allow-any-gpu`. The check will pass, but the
memory and timing figures in the notebooks were measured on a T4 and may not hold.

```bash
pytest tests/ -q
```

All tests should pass. This takes about a minute; most of it is a three-step GRPO run on
the GPU, which is there to catch a broken install before you find out in front of a class.
Without a GPU the GPU tests skip and the rest still run.

---

## 4. Running the notebooks

Open them from `notebooks/` in JupyterLab and run top to bottom. The working directory
matters - the notebooks add the repository root to `sys.path` relative to their own
location - and JupyterLab gets this right automatically.

- `01_classics_solutions.ipynb` - dynamic programming, Baird's counterexample, DQN.
- `02_grpo_solutions.ipynb` - GRPO in five acts.

The `_student.ipynb` versions are the same documents with the exercises blanked. They are
**build artefacts**: never edit one by hand, because `tools/make_student.py` regenerates
them from the solutions and will discard your edits.

### Two flags, both in the first cells

```python
SHARED_DIR = None          # or "/eos/project/.../rl-practice"
```

Where the large files live. `None` means "use this repository". If you were given an EOS
path, put it here - you do not need to set any environment variable or restart the kernel.
It is checked immediately, so a mount that is not up fails now rather than forty minutes in.

```python
TRAIN_FROM_SCRATCH = True  # False loads the pre-staged runs instead
```

The insurance policy. `False` skips the training and loads reference adapters and logs, and
every plot, evaluation and exercise below still works. Flip it if a run diverges, if the
GPU misbehaves, or if you are out of time. Rehearse this path at least once before teaching
from the notebook.

---

## 5. For instructors: pre-staging

Put a copy of the artefacts on a shared read-only volume - at CERN, EOS - laid out exactly
like the repository's `assets/`:

```
/eos/project/.../rl-practice/
    base_model/            <- the ~1 GB that is not in git; the only one that matters
    reference_adapters/    <- optional, the repository already ships these
    reference_logs/
    snapshots/
    dqn/
```

### What to copy, exactly

Everything below already exists in the repository except `base_model`, which is the only
thing that genuinely has to be staged. Copy the whole `assets/` directory and you are done:

| from the repository | size | needed for |
|---|---|---|
| `assets/base_model/` | **954 MB** | **everything in notebook 2** - the only mandatory one |
| `assets/reference_adapters/act1/`, `act4/` | 17 MB each | `TRAIN_FROM_SCRATCH = False` in notebook 2 |
| `assets/reference_logs/` | 48 KB | the same fallback: metrics, and sampled completions |
| `assets/snapshots/` | 12 KB | the same fallback: before/after evaluations |
| `assets/dqn/` | 320 KB | `TRAIN_FROM_SCRATCH = False` in notebook 1 |

```bash
cp -r rl-practice/assets/. /eos/project/.../rl-practice/
```

If you are short of space or time, `base_model/` alone is enough: the other four are
already in git, and students get them by cloning.

Build `base_model` on any machine with network access:

```bash
python tools/prestage.py --model            # just the weights, a few minutes
python tools/prestage.py --all              # weights, DQN runs, both GRPO reference runs
```

`--all` takes roughly 45 minutes on a T4 and is only needed if you want to regenerate the
reference runs; the repository already contains them.

Then tell students the path. Either they set `SHARED_DIR` in the first cell, or you set it
for the whole class on the notebook server spec (Advanced options, environment variables):

```
RLPRACTICE_SHARED_DIR=/eos/project/.../rl-practice
```

with no notebook edits at all. Nothing ever writes to this directory: all output goes to
the student's own checkout.

If the weights are somewhere that does not follow the `assets/` layout, point straight at
them instead with `RLPRACTICE_MODEL_DIR=/eos/.../whatever`.

---

## 6. If it goes wrong

**`torch.cuda.is_available()` is False.** Either the server was created without a GPU - it
cannot be added afterwards, make a new one - or pip replaced torch. Check with
`python tools/check_env.py`; if the torch version is not 2.3.1, reinstall with the
constraints file.

**Out of memory during notebook 2.** Peak is 7.1 GB of 15.6 on a T4, so this normally means
something else is on the GPU. Restart the kernel. If you are on a smaller card, reduce
`max_completion_length` in `rlpractice/grpo.py` before touching anything else - it is the
largest single lever on peak memory, because the logits tensor is
`batch x sequence x 152,000 vocabulary`.

**Cannot download the model.** You will get an error naming every path that was searched
and the three ways to fix it. The usual answer is `SHARED_DIR`.

**`ModuleNotFoundError: No module named 'rlpractice'`.** You are running the notebook from
the wrong directory. Open it from `notebooks/`, not from the repository root.

**The loss is zero on every step of notebook 2.** That is correct and not a bug. GRPO
advantages are mean-centred within each group by construction, so the reported loss is
approximately zero on every step of a healthy run. Watch the reward and the KL panel
instead; Act 3 is about exactly this.

**A training run collapses to gibberish.** Also not a bug, and it is roughly a one-in-four
event in Act 1 - with the KL penalty set to zero, nothing constrains how far the policy
travels from the model it started as. It is the point of Act 4. Set
`TRAIN_FROM_SCRATCH = False` and carry on with the analysis.
