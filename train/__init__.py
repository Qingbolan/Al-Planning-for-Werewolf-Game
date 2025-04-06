"""
Werewolf Game Training Module
"""

import sys
import os

# Get current module directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root directory path
parent_dir = os.path.dirname(current_dir)
# Add project root directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import RLTrainer class, handle possible import errors
try:
    from train.rl_trainer import RLTrainer
    HAS_RL_TRAINER = True
except (ImportError, ModuleNotFoundError):
    # If import fails, create placeholder class
    HAS_RL_TRAINER = False
    class RLTrainer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("RLTrainer module not implemented or missing dependencies")

__all__ = ['RLTrainer', 'HAS_RL_TRAINER']
