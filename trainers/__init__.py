"""Training Plane — executes training experiments from compiled Recipe IR configurations."""

from trainers.base import BaseTrainer
from trainers.distill import DistillTrainer
from trainers.sft import SFTTrainer
from trainers.rl import RLTrainer
from trainers.registry import get_trainer_class, register, list_registered

__all__ = [
    "BaseTrainer",
    "SFTTrainer",
    "RLTrainer",
    "DistillTrainer",
    "get_trainer_class",
    "register",
    "list_registered",
]
