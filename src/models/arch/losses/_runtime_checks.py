from __future__ import annotations

import torch


def tensor_value_checks_enabled() -> bool:
    """Skip tensor-data assertions while torch.compile is tracing graphs."""

    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return not bool(compiler.is_compiling())

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return not bool(dynamo.is_compiling())
    return True
