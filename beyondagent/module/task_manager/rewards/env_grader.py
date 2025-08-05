from typing import cast
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.schema.trajectory import Trajectory

from . import grader_manager

@grader_manager.reg("env")
class EnvGrader(RewardCalculator):
    def __init__(self):
        pass
    
    def set_instance_id(self,id:str):
        self._instance_id=id
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient) -> float:
        score = env.evaluate(self._instance_id, params={"sparse": True})
        return score