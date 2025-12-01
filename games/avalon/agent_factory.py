# -*- coding: utf-8 -*-
"""Agent factory for creating agents with role-specific configurations."""
from typing import Any, Callable, Optional

from agentscope.agent import AgentBase, ReActAgent
from agentscope.model import ModelBase

from tutorial.example_avalon.agents.thinking_react_agent import ThinkingReActAgent


def create_agent_for_role(
    player_id: int,
    role_name: str,
    model_factory: Optional[Callable[[], ModelBase]] = None,
    agent_class: type[ReActAgent] = ThinkingReActAgent,
    agent_kwargs: Optional[dict[str, Any]] = None,
) -> ReActAgent:
    """Create an agent for a specific role.
    
    Args:
        player_id: Player ID (0-indexed).
        role_name: Role name (e.g., "Merlin", "Servant", "Assassin").
        model_factory: Optional callable that returns a ModelBase instance.
                      If None, agent will be created without a model (for UserAgent).
        agent_class: Agent class to instantiate. Defaults to ThinkingReActAgent.
        agent_kwargs: Additional keyword arguments for agent initialization.
    
    Returns:
        An instance of the specified agent class.
    """
    agent_kwargs = agent_kwargs or {}
    name = f"Player{player_id}"
    
    if model_factory:
        model = model_factory()
        return agent_class(
            name=name,
            sys_prompt="",  # System prompt will be set in game.py
            model=model,
            **agent_kwargs
        )
    else:
        # For UserAgent or agents without model
        return agent_class(name=name, **agent_kwargs)


def create_agents_with_role_config(
    num_players: int,
    role_model_config: Optional[dict[str, Callable[[], ModelBase]]] = None,
    default_model_factory: Optional[Callable[[], ModelBase]] = None,
    agent_class: type[ReActAgent] = ThinkingReActAgent,
    agent_kwargs: Optional[dict[str, Any]] = None,
) -> list[ReActAgent]:
    """Create agents with role-specific model configurations.
    
    Args:
        num_players: Number of players in the game.
        role_model_config: Dictionary mapping role names to model factories.
                         Example: {"Merlin": lambda: Model1(), "Assassin": lambda: Model2()}
        default_model_factory: Default model factory for roles not in role_model_config.
        agent_class: Agent class to instantiate. Defaults to ThinkingReActAgent.
        agent_kwargs: Additional keyword arguments for agent initialization.
    
    Returns:
        List of agents. Note: Roles are assigned later by the game environment,
        so this creates agents with placeholder configurations.
    """
    agents = []
    for i in range(num_players):
        # For now, use default model factory since roles are assigned later
        # In the future, we could support pre-assigning roles
        model_factory = default_model_factory
        agent = create_agent_for_role(
            player_id=i,
            role_name="Unknown",  # Will be assigned by game
            model_factory=model_factory,
            agent_class=agent_class,
            agent_kwargs=agent_kwargs,
        )
        agents.append(agent)
    return agents


def update_agents_models_by_roles(
    agents: list[ReActAgent],
    roles: list[tuple],
    role_model_config: dict[str, Callable[[], ModelBase]],
    default_model_factory: Optional[Callable[[], ModelBase]] = None,
) -> None:
    """Update agents' models based on their assigned roles.
    
    Args:
        agents: List of agents to update.
        roles: List of (role_id, role_name, side) tuples from game environment.
        role_model_config: Dictionary mapping role names to model factories.
        default_model_factory: Default model factory if role not in config.
    """
    for i, (_, role_name, _) in enumerate(roles):
        if i < len(agents) and hasattr(agents[i], 'model'):
            model_factory = role_model_config.get(role_name, default_model_factory)
            if model_factory:
                agents[i].model = model_factory()

