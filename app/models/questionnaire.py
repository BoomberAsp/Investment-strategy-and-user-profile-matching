"""
数据模型：Question, Questionnaire
"""

from dataclasses import dataclass


@dataclass
class Question:
    q_id: str
    text: str
    q_type: str                 # "single_choice" | "slider" | "number_input" | "multi_select"
    options: list[str] = None   # 单选/多选选项
    target_params: list[str] = None  # 影响的超参数
    weights: dict = None        # 各选项对 target 的映射权重

    def __post_init__(self):
        if self.options is None:
            self.options = []
        if self.target_params is None:
            self.target_params = []
        if self.weights is None:
            self.weights = {}


@dataclass
class Questionnaire:
    level: str                  # "L1" | "L2" | "L3"
    title: str
    description: str
    questions: list[Question]
    estimated_minutes: int
