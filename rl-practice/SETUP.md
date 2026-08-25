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

Two things to get: the code, from git, and the data, from a download link. Neither is much
use without the other.

Open a terminal in JupyterLab (File, New, Terminal) and run:

```bash
# 1. the code
git clone https://github.com/CERN-STEAM-Academy/26-ADVANCED-ML.git
cd 26-ADVANCED-ML/rl-practice

# 2. the python dependencies
pip install --user -c constraints.txt -r requirements.txt

# 3. the data: 782 MB, and it unpacks into a directory called assets/
#    NOTE the URL form: /s/<token>/download   (see the warning below)
cd ~
wget -O assets.tar.gz "https://cernbox.cern.ch/s/QbQHtpgOSgkpCho/download"
tar xzf assets.tar.gz
```

Then **tell the notebooks where you unpacked it**. In the first cell of either notebook:

```python
SHARED_DIR = "~/assets"
```

That is the whole setup. No environment variables, no kernel restart.

### The download URL has a trap in it

The share page you may have been sent looks like this, and opens fine in a browser:

```
https://cernbox.cern.ch/index.php/s/QbQHtpgOSgkpCho
```

but appending `/download` to *that* form does not work. CERNBox answers it with a
redirect to `https://cernbox.cern.chs/...` - note the stray `s` on the hostname - and
`wget` dies with "Could not resolve host". Verified, not guessed.

Use the short form instead, which returns the archive directly:

```
https://cernbox.cern.ch/s/QbQHtpgOSgkpCho/download
```

### Getting `SHARED_DIR` right

It must point at the directory that **directly contains `base_model/`**. This archive
unpacks to `assets/`, so if you unpacked it in your home directory, that is `~/assets`:

```bash
ls ~/assets      # base_model  dqn  reference_adapters  reference_logs  snapshots
```

If you point it one level too high or too low, the notebook stops immediately and tells you
what it found, what it expected, and usually which directory you actually meant. It does
not fail quietly forty minutes later. The first cell prints where every artefact was
resolved from, so read that output once.

**Shortcut:** unpack the archive inside the repository instead, and there is nothing to
configure at all - the archive's `assets/` lands exactly where the notebooks look by
default, so `SHARED_DIR` can stay `None`:

```bash
cd 26-ADVANCED-ML/rl-practice && tar xzf ~/assets.tar.gz
```

### About `pip install --user`

`--user` because the image's `site-packages` is root-owned and you cannot write to it.
Installs land in `~/.local`, which is on your persistent volume and survives a restart.

**Do not drop the `-c constraints.txt`.** It pins torch, torchvision, torchaudio and numpy
to what the image already ships. Without it, one of the packages being installed can decide
it needs a newer torch, and pip will quietly replace the CUDA build the image was tested
with. That surfaces much later as `torch.cuda.is_available()` returning False, or kernels
that do not exist for this GPU, and it is miserable to diagnose. With the constraints file,
the same situation is an install-time resolver error instead.

The image also sets `PIP_CONSTRAINT=/etc/pip-constraints.txt` (pinning `numpy==1.24.4`).
Passing `-c` on the command line does **not** override that - pip applies both files. This
was verified rather than assumed.

### If you skip the download

You can, for notebook 1: it is CPU-only and generates everything it uses, so it runs from a
bare clone. Notebook 2 cannot - it needs the model weights. With `SHARED_DIR = None` it
falls back to downloading them from the HuggingFace hub on first use, which needs network
access and takes a few minutes.

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
SHARED_DIR = None          # or "~/assets", or "/eos/project/.../assets"
```

Where you unpacked `assets.tar.gz`. `None` means "look inside this repository", which is
correct if you unpacked it there. You do not need to set an environment variable or restart
the kernel, and the path is checked immediately, so a wrong one fails now rather than forty
minutes in.

```python
TRAIN_FROM_SCRATCH = True  # False loads the pre-staged runs instead
```

The insurance policy. `False` skips the training and loads reference adapters and logs, and
every plot, evaluation and exercise below still works. Flip it if a run diverges, if the
GPU misbehaves, or if you are out of time. Rehearse this path at least once before teaching
from the notebook.

---

## 5. For instructors: building assets.tar.gz

Nothing under `assets/` is in git - not the model, and not the reference runs. Weights are
not source, and 34 MB of binary adapters bloats every clone forever. It is all distributed
as one archive instead.

### What goes in it

| directory | size | needed for |
|---|---|---|
| `base_model/` | **954 MB** | **everything in notebook 2** - the only mandatory one |
| `reference_adapters/act1/`, `act4/` | 17 MB each | `TRAIN_FROM_SCRATCH = False`, notebook 2 |
| `reference_logs/` | 48 KB | the same fallback: metrics and sampled completions |
| `snapshots/` | 12 KB | the same fallback: before/after evaluations |
| `dqn/` | 252 KB | `TRAIN_FROM_SCRATCH = False`, notebook 1 |
| | **~990 MB** | |

### Building it

On any machine with network access and a GPU:

```bash
python tools/prestage.py --all        # ~45 min on a T4: weights, DQN runs, both GRPO runs
```

or, if you only need the weights and are happy with the reference runs already produced:

```bash
python tools/prestage.py --model      # a few minutes, just the ~1 GB of weights
```

Then pack it, keeping `assets` as the top-level directory so that unpacking it inside the
repository is a no-configuration install:

```bash
cd rl-practice
tar czf assets.tar.gz assets        # 782 MB
```

Upload it to CERNBox and make the link public. Then give students the **short-form**
download URL, `https://cernbox.cern.ch/s/<token>/download` - the `/index.php/s/<token>`
form redirects to a malformed hostname when `/download` is appended and will not work with
`wget` or `curl`.

### Or use a shared filesystem instead

If every student has the same mount - EOS, for instance - skip the download entirely. Put
the unpacked directory somewhere readable and set it once on the notebook-server spec
(Advanced options, environment variables):

```
RLPRACTICE_SHARED_DIR=/eos/project/.../rl-practice-assets
```

Students then do nothing at all: no download, no `SHARED_DIR` to edit. Nothing ever writes
to that directory; all output goes to the student's own checkout.

If the weights are somewhere that does not follow this layout, point straight at them with
`RLPRACTICE_MODEL_DIR=/eos/.../whatever` instead.

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
and the ways to fix it. The usual answer is that `SHARED_DIR` is not set, or is set to the
wrong level of the unpacked archive - it must point at the directory that directly contains
`base_model/`.

**`SHARED_DIR` is set but the notebook says it does not contain `base_model/`.** You have
pointed at the directory you unpacked *into* rather than the one the archive created. The
error message suggests the right one; use that.

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
