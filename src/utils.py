import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "repo", "original_architecture_reproduction")))

from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
