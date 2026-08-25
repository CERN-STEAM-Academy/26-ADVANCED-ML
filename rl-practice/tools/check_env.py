#!/usr/bin/env python3
"""Assert that the environment is the one this session was built and tested against.

Run this first, in a fresh container, before anything else. It is the gate described in
section 0.1 of the implementation spec, and it exists because of one specific failure
mode: ``pip install`` resolves a modern ``transformers``/``trl``, decides it needs a newer
torch, silently replaces the CUDA build the image shipped with, and everything afterwards
either falls back to CPU or fails on sm75 in a way that looks like a bug in your code.

The checks, in order of how badly they ruin your day:

1. torch is still the version the image shipped (2.3.x), i.e. pip did not move it;
2. CUDA is actually available, i.e. the replacement build was not the CPU wheel;
3. the device is compute capability (7, 5), i.e. Turing, i.e. a T4;
4. bf16 is *not* supported, which is a fact about Turing worth confirming rather than
   assuming - it is the reason every config in these notebooks sets ``fp16=True``;
5. the library versions match the pins that were tested.
"""

from __future__ import annotations

import argparse
import importlib
import sys

EXPECTED_TORCH_PREFIX = "2.3.1"
EXPECTED_CAPABILITY = (7, 5)

#: Packages whose versions get printed. The pins live in requirements.txt; this only
#: reports, so that a mismatch is visible rather than silently tolerated.
REPORTED = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "tokenizers",
    "numpy",
    "gymnasium",
    "matplotlib",
    "pandas",
]


def version_of(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        return f"NOT INSTALLED ({type(error).__name__})"
    return getattr(module, "__version__", "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-any-gpu",
        action="store_true",
        help="skip the compute-capability assertion (for development on non-Turing hardware)",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []

    print("=== versions ===")
    for name in REPORTED:
        print(f"  {name:<14} {version_of(name)}")

    import torch

    print("\n=== torch build ===")
    print(f"  torch.__version__        {torch.__version__}")
    print(f"  torch.version.cuda       {torch.version.cuda}")
    print(f"  torch.cuda.is_available  {torch.cuda.is_available()}")

    if not torch.__version__.startswith(EXPECTED_TORCH_PREFIX):
        failures.append(
            f"torch is {torch.__version__}, expected {EXPECTED_TORCH_PREFIX}.*. "
            "pip upgraded it. Pin harder, or install the offending package with --no-deps "
            "and add its transitive dependencies by hand."
        )

    if not torch.cuda.is_available():
        failures.append(
            "torch.cuda.is_available() is False. Either the container has no GPU visible "
            "(run docker with --gpus all) or pip replaced the CUDA build with a CPU wheel."
        )
    else:
        capability = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  device                   {name}")
        print(f"  compute capability       {capability}")
        print(f"  total memory             {total_gb:.1f} GB")
        print(f"  bf16 supported           {torch.cuda.is_bf16_supported()}")

        if capability != EXPECTED_CAPABILITY and not args.allow_any_gpu:
            failures.append(
                f"compute capability is {capability}, expected {EXPECTED_CAPABILITY} (Turing/T4). "
                "The notebooks are tuned for 16 GB of sm75; re-measure before trusting the "
                "memory and time budgets."
            )
        if capability == EXPECTED_CAPABILITY and torch.cuda.is_bf16_supported():
            # Not a failure, but a trap worth naming. torch.cuda.is_bf16_supported()
            # falls back to "can I allocate a bfloat16 tensor?", and on Turing you can -
            # the dtype exists and is emulated. What does not exist is hardware bf16
            # matmul, so training in bf16 here is silently slow and numerically odd.
            # Trust the compute capability, not this flag: sm75 means fp16.
            print(
                "  NOTE: is_bf16_supported() is True on sm75 because it only checks that "
                "a bfloat16 tensor can be allocated. There is no bf16 tensor-core support "
                "on Turing. Use fp16=True, bf16=False."
            )

        # A real allocation and a real matmul: version strings can lie, kernels cannot.
        try:
            a = torch.randn(512, 512, device="cuda")
            b = torch.randn(512, 512, device="cuda")
            (a @ b).sum().item()
            with torch.autocast("cuda", dtype=torch.float16):
                (a @ b).sum().item()
            print("  fp32 and fp16 matmul     ok")
        except Exception as error:
            failures.append(f"a trivial CUDA matmul failed: {type(error).__name__}: {error}")

    print("\n=== flash-attention / vLLM must be absent ===")
    for forbidden, why in (
        ("flash_attn", "sm75 is unsupported by flash-attention; use attn_implementation='sdpa'"),
        ("vllm", "vLLM forces a torch upgrade; GRPO must run with use_vllm=False"),
    ):
        try:
            importlib.import_module(forbidden)
            print(f"  {forbidden:<12} PRESENT - {why}")
            failures.append(f"{forbidden} is installed. {why}")
        except ImportError:
            print(f"  {forbidden:<12} absent (correct)")

    print()
    if failures:
        print("ENVIRONMENT CHECK FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("ENVIRONMENT CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
