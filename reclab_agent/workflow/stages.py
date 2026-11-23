"""The four business stages; execution strategies are independent of stages."""
from enum import Enum


class StudyStage(str, Enum):
    BACKGROUND = "background"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    ANALYSIS = "analysis"
