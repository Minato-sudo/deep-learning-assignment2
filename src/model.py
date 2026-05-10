import os
import sys

# Append the original repo to path to ensure all dependencies work as before
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "repo", "original_architecture_reproduction")))

from model import UNet
