# -*- coding: utf-8 -*-
"""Diplomacy workflows package."""

from games.diplomacy.workflows.rollout_workflow import DiplomacyWorkflow
from games.diplomacy.workflows.eval_workflow import EvalDiplomacyWorkflow

__all__ = [
    "DiplomacyWorkflow",
    "EvalDiplomacyWorkflow",
]
