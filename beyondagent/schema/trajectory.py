from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Reward(BaseModel):
    outcome: float = Field(default=0.0)
    description: str = Field(default="Outcome 1 denotes success, and 0 denotes failure.")

    metadata: dict = Field(default_factory=dict)


class Trajectory(BaseModel):
    data_id: str = Field(default="")
    rollout_id: str = Field(default="")

    steps: List[Dict[str, str]] = Field(default_factory=list)
    query: str = Field(default="")

    is_terminated: bool = Field(default=False)
    reward: Reward = Field(default_factory=Reward)

    metadata: dict = Field(default_factory=dict)


class Sample(BaseModel):
    """The data model for single sample."""

    data_id: int = 0
    rollout_id: int = 0
    messages: List[Dict[str, Any]] = []
    extras: Dict[str, Any] = {}
    input_ids: List[int] = None
    prompt_ids: List[int] = None
    response_ids: List[int] = None
    attention_mask: List[int] = None
    prompt_attention_mask: List[int] = None
    response_attention_mask: List[int] = None
    position_ids: List[int] = None
    prompt_position_ids: List[int] = None
    response_position_ids: List[int] = None
    loss_mask: List[int] = None
    prompt_loss_mask: List[int] = None
    response_loss_mask: List[int] = None
    reward_scores: Dict[str, Any] = None

    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.response_ids[: self.max_response_len]
        self.response_attention_mask = self.response_attention_mask[: self.max_response_len]
        self.response_position_ids = self.response_position_ids[: self.max_response_len]
        self.response_loss_mask = self.response_loss_mask[: self.max_response_len]

    def discard(self) -> None:
        """
        Discard the experience.
        """
        self.input_ids = []
        self.position_ids = []
        self.attention_mask = []
        self.loss_mask = []
        self.prompt_ids = []
        self.response_ids = []
        self.prompt_attention_mask = []
        self.response_attention_mask = []
        self.prompt_position_ids = []
        self.response_position_ids = []
        self.prompt_loss_mask = []
        self.response_loss_mask = []
