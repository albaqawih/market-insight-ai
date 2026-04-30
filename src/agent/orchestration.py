"""
Multi-agent orchestration layer.

Design and implement your own orchestration strategy here.
This file is entirely yours — there is no single correct answer.
"""

from src.agent.base import BaseAgent
from src.agent.prompts import (
    ANALYST_PROMPT,
    RESEARCHER_PROMPT,
    WRITER_PROMPT,
)
from src.config import settings


class OrchestratorAgent:
    """Minimal sequential pipeline: Researcher -> Analyst -> Writer."""

    def __init__(self, model: str = None, max_steps: int = 10):
        resolved_model = model or settings.model_name
        self.researcher = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Researcher",
            system_prompt=RESEARCHER_PROMPT,
        )
        self.analyst = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Analyst",
            system_prompt=ANALYST_PROMPT,
        )
        self.writer = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Writer",
            system_prompt=WRITER_PROMPT,
        )

    async def run(self, query: str) -> dict:
        research_output = await self.researcher.run(query)
        analysis_output = await self.analyst.run(research_output["answer"])
        final_output = await self.writer.run(analysis_output["answer"])
        return final_output
