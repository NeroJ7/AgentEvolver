from typing import cast
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.schema.trajectory import Trajectory

USER_PROMPT="""Based on the conversation trajectory above, evaluate the task completion quality using the framework provided.

Your evaluation should address the following dimensions in order:

**Step 1: Relevance Check (0 or proceed)**
- Are the solution steps relevant to the problem? If the approach is completely unrelated to the task requirements, assign 0 points immediately.
- If relevant, proceed to other evaluation dimensions.

**Step 2: Final Answer Assessment (Critical)**
- Does the agent provide a final answer? Is the final answer correct?
- No answer or incorrect answer: Maximum 49 points
- Correct answer provided: Minimum 60 points

**Step 3: Solution Efficiency**
- Are there unnecessary or irrelevant steps in the solution process?
- Deduct points for redundant or off-topic actions
- If the final answer is correct, the minimum score after deductions should not go below 60 points

**Step 4: Repetition Penalty**
- Does the agent get stuck in infinite loops or repeat identical steps endlessly?
- If there are infinite repetitions of the same steps, even with a correct answer, the maximum score is 20 points

**Step 5: Code Execution Quality**
- Are there code execution errors during the process?
- Deduct points appropriately for runtime errors, bugs, or failed executions

**Scoring Guidelines:**
- 90-100: Exceptional performance - correct answer with efficient, clean execution
- 80-89: Strong performance - correct answer with minor inefficiencies
- 70-79: Good performance - correct answer with some unnecessary steps
- 60-69: Adequate performance - correct answer but with notable issues
- 50-59: Poor performance - major issues but some progress made
- 20-49: Very poor performance - no correct answer or severe execution problems
- 1-19: Minimal attempt with infinite loops or severe repetition issues
- 0: Complete failure - irrelevant approach or no meaningful attempt

Provide your detailed analysis first, explaining your reasoning for each evaluation dimension. Then assign a precise integer score between 0 and 100 based on the criteria above.

First provide your detailed reasoning analysis, then output an integer score between 0 and 100 enclosed in <reward></reward> tags, e.g., <reward>75</reward>
"""
# query & reward improvement
# TODO 可以与 appworld grader 算【相关性】
# 非 sparse reward 对 llm 的要求会比较低
# use 0～100
# 要试试把 reference traj 拿过来吗

class LlmAsJudgeRewardCalculator(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge.
    
    TODO: This is a temperary solution for synthetic data.
    """
    def __init__(self, model_name='qwen-plus'):
        self._client=DashScopeClient(model_name=model_name)
    
    def pack_message(self, trajectory: Trajectory):
        """Pack trajectory into a message.
        
        Args:
            trajectory (Trajectory): trajectory to pack
        """
        messages=[]
        
        # 添加轨迹消息（将所有对话转换为一个连贯的文本）
        trajectory_text = "The following is the dialogue trace of the task execution:\n\n"
        for i, msg in enumerate(trajectory.steps):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            trajectory_text += f"{role.upper()}: {content}\n\n"
        
        messages.append({"role": "user", "content": trajectory_text})
        messages.append({"role":"user","content":USER_PROMPT})
        return messages
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient) -> float:
        x=cast(float,self._calculate_reward(trajectory,env,eject_llm_output=False))
        return x
        

    def _calculate_reward(self, trajectory: Trajectory, env:EnvClient, *, eject_llm_output:bool=False):
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        response=""
        for chunk in self._client.chat_stream_with_retry(messages=self.pack_message(trajectory)):
            response += chunk
        if response:
            import re
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score = float(reward_match.group(1))
                score = max(0.0, min(100.0, score))/100.0
            else:
                print(f"Could not parse score from response: {response}")
                score=0.0
        else:
            print("No response from evaluation API")
            score=0.0
        
        if not eject_llm_output:
            return score
        else:
            return score,response