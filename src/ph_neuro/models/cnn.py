"""Pre-built Hebbian CNN model placeholder.

Convolutional Hebbian network for vision tasks (Phase 1+).
"""

from __future__ import annotations

import torch.nn as nn


class HebbianCNN(nn.Module):
    """Hebbian CNN placeholder for Phase 1.

    This will be implemented when ``TernaryHebbianConv2d`` is available.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError(
            "HebbianCNN is not yet implemented. It will be available in Phase 1 (Vision POC)."
        )
