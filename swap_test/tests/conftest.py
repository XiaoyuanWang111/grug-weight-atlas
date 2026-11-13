"""Shared test settings: two torch threads and a fixed global seed."""

import torch

torch.set_num_threads(2)
torch.manual_seed(0)
