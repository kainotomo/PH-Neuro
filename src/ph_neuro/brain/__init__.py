"""PH-Neuro Brain — local surprise-modulated plasticity on frozen LMs.

Phase 1.1 (E031) minimal-viable implementation of the locked Step 0.4
architecture: wrap any HuggingFace ``AutoModelForCausalLM`` with tiny vector
biases injected at each block's output projections via output-modification
forward hooks, updated by a global surprise-modulated 3-factor Hebbian rule
(``Δb = η·M·mean(post)``) with **no backpropagation**.

Public surface::

    from ph_neuro.brain import BrainWrapper, SurpriseModulator, get_block_wrapper
"""

from __future__ import annotations

from ph_neuro.brain.block_wrappers import (
    GPT2BlockWrapper,
    InjectionPoint,
    SmolLM2BlockWrapper,
    get_block_container,
    get_block_wrapper,
)
from ph_neuro.brain.brain_wrapper import BrainWrapper
from ph_neuro.brain.modulator import SurpriseModulator

__all__ = [
    "BrainWrapper",
    "SurpriseModulator",
    "InjectionPoint",
    "BlockWrapper",
    "SmolLM2BlockWrapper",
    "GPT2BlockWrapper",
    "get_block_wrapper",
    "get_block_container",
]

# Imported for re-export typing only (Protocol re-export convenience).
from ph_neuro.brain.block_wrappers import BlockWrapper as BlockWrapper  # noqa: E402,F401,PLC0415
