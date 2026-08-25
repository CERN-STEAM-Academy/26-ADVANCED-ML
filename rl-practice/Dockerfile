# Image for the RL practice session. Build once, and the students' setup is `git clone`.
#
#   docker build -t registry.cern.ch/ngt/rl-practice:2026 -f rl-practice/Dockerfile .
#   docker push registry.cern.ch/ngt/rl-practice:2026
#
# Why this base rather than registry.cern.ch/ngt/pytorch:2.3.1, which already has torch:
#
#   1. That image has 153 layers. Docker's overlay2 storage driver caps at 128, so it
#      cannot be pulled or built FROM on an ordinary Docker host at all ("max depth
#      exceeded"). This one has 24.
#   2. Its conda scipy is broken: `_highs_wrapper.so` needs GLIBCXX_3.4.30 and the system
#      libstdc++ is older, so `import sklearn` raises. transformers imports sklearn behind
#      an availability check that tests whether the package is *installed* rather than
#      whether it *imports*, so the guard passes and the import explodes. Measured: four
#      GPU tests died that way. This base is miniforge/conda-forge and `scipy.optimize`
#      imports cleanly.
#   3. Installing our dependencies over EOS-backed home took 24 minutes per student.
#      Baking them in makes it zero.
#
# What this adds: torch 2.3.1+cu121 (~2.5 GB of CUDA wheels) plus about 300 MB of the
# session's own dependencies.
FROM registry.cern.ch/ngt/scipy:master

USER root

# Pin the things that must not move. torch is pinned because a later resolver step could
# otherwise replace the CUDA build; numpy is pinned below 2 because torch 2.3.1 predates
# the numpy 2 ABI break. numpy is left at whatever the base already has (1.26.4) rather
# than forced to an exact version - the base's numba, pandas and scipy are built against
# it, and moving it to satisfy a pin copied from a different image would break them.
RUN printf 'torch==2.3.1\nnumpy<2\n' > /etc/rl-practice-constraints.txt

# CUDA build of torch, from the PyTorch index. cu121 is the build that supports sm75
# (Turing, which is what a T4 is) and matches the CUDA runtime the notebooks were
# measured against.
#
# torch only - no torchvision, no torchaudio. Nothing in this session imports either, and
# leaving them out saves about a gigabyte in an image that is already large because of the
# bundled CUDA libraries.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      -c /etc/rl-practice-constraints.txt \
      torch==2.3.1

# The session's own dependencies.
COPY rl-practice/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -c /etc/rl-practice-constraints.txt -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

# Fail the build rather than the lecture. Each of these is a failure mode that actually
# happened during development.
RUN python - <<'PY'
import torch, transformers, trl, peft, accelerate, datasets, gymnasium
assert torch.__version__.startswith("2.3.1"), f"torch moved to {torch.__version__}"
assert torch.version.cuda == "12.1", f"wrong CUDA build: {torch.version.cuda}"
assert not torch.__version__.endswith("+cpu"), "CPU-only wheel installed"
import numpy as np
assert np.__version__.startswith("1."), f"numpy {np.__version__} breaks torch 2.3.1"
# The exact chain that fails on the pytorch base: transformers guards its sklearn import
# with a check that sklearn is *installed*, not that it *imports*.
import sklearn, scipy.optimize                                   # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: F401
from trl import GRPOTrainer                                      # noqa: F401
print("build check OK:", torch.__version__, "| transformers", transformers.__version__,
      "| trl", trl.__version__, "| numpy", np.__version__)
PY

# Turing has no bf16 tensor cores; the notebooks use fp16 throughout. Recorded here so the
# reason travels with the image.
# Ignore ~/.local/lib/python*/site-packages entirely.
#
# On Kubeflow the home directory is a persistent volume - at CERN, EOS - and it outlives
# any one notebook server. A `pip install --user` done once, against some other image,
# leaves packages there that then take priority over this image's own on every future
# server. Observed: a numpy left in EOS home by an earlier install shadowed the image's
# numpy and broke `import matplotlib` with a circular-import error that names neither the
# stale directory nor the real cause.
#
# This image is self-contained, so user site-packages can only do harm. Verified: with a
# stale numpy in ~/.local, imports fail; with this set, numpy resolves to /opt/conda and
# matplotlib imports cleanly.
ENV PYTHONNOUSERSITE=1

# Matplotlib wants a writable config directory and warns loudly on every import when HOME
# is read-only or unwritable.
ENV MPLCONFIGDIR=/tmp/matplotlib

LABEL org.opencontainers.image.description="CERN STEAM Academy 2026 - RL practice session. \
torch 2.3.1+cu121 for NVIDIA T4 (sm75), transformers/trl/peft for GRPO, gymnasium for DQN."

USER ${NB_USER}
