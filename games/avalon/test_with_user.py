# -*- coding: utf-8 -*-
"""Example test script with role-specific model configuration and UserAgent support."""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

# Add astune directory to path for imports
astune_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(astune_dir))

from agentscope.agent import UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
from games.avalon.game import avalon_game
from games.avalon.engine import AvalonBasicConfig


async def main(language: str = "en", use_user_agent: bool = False, user_agent_id: int = 0):
    """Main function to run Avalon game with role-specific configurations.
    
    Args:
        language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
        use_user_agent: Whether to use UserAgent for one player.
        user_agent_id: Player ID to use UserAgent (0-indexed).
    """
    num_players = 5
    config = AvalonBasicConfig.from_num_players(num_players)
    
    # Model configuration
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    api_key = os.getenv("API_KEY", "sk-224e008372e144e496e06038077f65fc")
    
    lang_display = "中文" if language.lower() in ["zh", "cn", "chinese"] else "English"
    print(f"Initializing Avalon game with {num_players} players...")
    print(f"Language: {lang_display}")
    print(f"Model: {model_name}")
    if use_user_agent:
        print(f"UserAgent enabled for Player{user_agent_id}")
    print()
    
    # Define model factories for different roles
    def create_default_model() -> DashScopeChatModel:
        """Default model factory."""
        return DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=False,
        )
    
    def create_merlin_model() -> DashScopeChatModel:
        """Model factory for Merlin (can use different model/params)."""
        return DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=False,
            # Can customize temperature, max_tokens, etc. here
        )
    
    def create_assassin_model() -> DashScopeChatModel:
        """Model factory for Assassin (can use different model/params)."""
        return DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=False,
        )
    
    # Create agents
    # Note: Roles are assigned randomly by the game, so we create agents with default models
    # For role-specific models, you would need to:
    # 1. Pre-assign roles, or
    # 2. Create agents after role assignment and swap models accordingly
    agents = []
    for i in range(num_players):
        if use_user_agent and i == user_agent_id:
            # Create UserAgent for interactive play
            agent = UserAgent(name=f"Player{i}")
            print(f"Created {agent.name} (UserAgent - interactive)")
        else:
            # Create ThinkingReActAgent with model
            # You can customize models per player here if needed
            model = create_default_model()
            agent = ThinkingReActAgent(
                name=f"Player{i}",
                sys_prompt="",  # System prompt will be set in game.py
                model=model,
                formatter=DashScopeChatFormatter(),
                memory=InMemoryMemory(),
                toolkit=Toolkit(),
            )
            print(f"Created {agent.name} (ThinkingReActAgent)")
        agents.append(agent)
    
    print()
    print("=" * 60)
    print("Game Starting...")
    print("=" * 60)
    print()
    
    # Run game with logging
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        good_wins = await avalon_game(agents, config, log_dir=log_dir, language=language)
        
        print()
        print("=" * 60)
        print("Game Finished!")
        print("=" * 60)
        print(f"Result: {'Good wins!' if good_wins else 'Evil wins!'}")
        print(f"Logs saved to: {log_dir}")
        print()
        
        return good_wins
    except Exception as e:
        print(f"Error during game: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Avalon game with role-specific configurations")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=os.getenv("LANGUAGE", "en"),
        choices=["en", "zh", "cn", "chinese"],
        help='Language for prompts: "en" for English, "zh"/"cn"/"chinese" for Chinese (default: en)',
    )
    parser.add_argument(
        "--use-user-agent",
        action="store_true",
        help="Use UserAgent for one player (interactive mode)",
    )
    parser.add_argument(
        "--user-agent-id",
        type=int,
        default=0,
        help="Player ID to use UserAgent (0-indexed, default: 0)",
    )
    args = parser.parse_args()
    
    asyncio.run(main(language=args.language, use_user_agent=args.use_user_agent, user_agent_id=args.user_agent_id))

