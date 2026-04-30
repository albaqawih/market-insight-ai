"""
Centralized prompts for all agents and planners in the system.

Note: These are example roles. You are encouraged to add, remove, or 
completely redesign these agents to fit your orchestration strategy.
"""

DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer clearly and concisely.
""".strip()

PLANNER_PROMPT = """TODO: Design a prompt that breaks complex queries into sub-tasks."""

RESEARCHER_PROMPT = """TODO: Design a prompt for a Research Specialist who finds and retrieves information."""

ANALYST_PROMPT = """TODO: Design a prompt for an Analyst who predicts stock movement from news."""

WRITER_PROMPT = """TODO: Design a prompt for a Writing Specialist who produces polished reports."""

FACT_CHECKER_PROMPT = """TODO: Design a prompt for a Fact Checker who verifies claims."""
