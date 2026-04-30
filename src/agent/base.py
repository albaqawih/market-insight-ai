"""
BaseAgent: a ReAct agent with decorator-based observability.

Observability uses the same @observe / propagate_attributes API as production Langfuse.
Swapping to real Langfuse requires only changing the import:
    from langfuse import observe, propagate_attributes
"""

import asyncio
import json
import os

import structlog
from litellm import acompletion, completion_cost
from pydantic import ValidationError

from src.agent.prompts import DEFAULT_SYSTEM_PROMPT
from src.config import settings
from src.observability.detectors import LoopDetector
from src.observability.observe import observe, propagate_attributes

logger = structlog.get_logger()


class BaseAgent:
    """
    A ReAct agent with full observability:
    - Decorator-based tracing of every call (@observe)
    - Loop detection (exact, fuzzy, stagnation)
    - Per-run cost tracking
    - Async execution
    """

    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 10,
        agent_name: str = "BaseAgent",
        verbose: bool = True,
        system_prompt: str | None = None,
        tools: list | None = None,
    ):
        self.model = model or settings.model_name
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.verbose = verbose


        self.tools = tools or []
        self.tools_schema = [tool.to_openai_schema() for tool in self.tools]
        self.loop_detector = LoopDetector()


    @observe(name="agent_run", as_type="agent")
    async def run(self, user_query: str) -> dict:
        with propagate_attributes(metadata={"user_query": user_query}):
            if not os.getenv("OPENAI_API_KEY"):
                answer = self._mock_answer(user_query)
                return {
                    "answer": answer,
                    "metadata": {
                        "agent_name": self.agent_name,
                        "model": self.model,
                        "total_steps": 1,
                        "total_cost_usd": 0.0,
                        "mock_mode": True,
                    },
                }

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
            ]
            max_iters = min(self.max_steps, 2)
            total_cost = 0.0

            for step in range(1, max_iters + 1):
                response = await acompletion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_schema or None,
                )
                total_cost += completion_cost(response)
                assistant_message = response.choices[0].message
                content = assistant_message.content or ""
                tool_calls = assistant_message.tool_calls or []

                if not tool_calls:
                    return {
                        "answer": content,
                        "metadata": {
                            "agent_name": self.agent_name,
                            "model": self.model,
                            "total_steps": step,
                            "total_cost_usd": round(total_cost, 6),
                        },
                    }

                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            tc.model_dump() if hasattr(tc, "model_dump") else tc
                            for tc in tool_calls
                        ],
                    }
                )

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        tool_args = {}

                    tool_result = await self._execute_tool(tool_name, tool_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": tool_result,
                        }
                    )

            return {
                "answer": "No final answer generated within 2 iterations.",
                "metadata": {
                    "agent_name": self.agent_name,
                    "model": self.model,
                    "total_steps": max_iters,
                    "total_cost_usd": round(total_cost, 6),
                },
            }

    def _mock_answer(self, user_query: str) -> str:
        name = self.agent_name.lower()
        if name == "researcher":
            return (
                "- Tesla announced stronger-than-expected quarterly deliveries.\n"
                "- A major broker upgraded Tesla outlook for next quarter.\n"
                "- EV demand remains stable in key US and EU markets.\n"
                "- Rising competition may pressure Tesla margins."
            )
        if name == "analyst":
            return (
                "Prediction: Slight Rise\n"
                "Reason: Most updates are positive, but competition risk limits stronger upside."
            )
        if name == "writer":
            return (
                "تشير الأخبار الأخيرة إلى صورة إيجابية معتدلة لسهم تسلا.\n"
                "أبرز العوامل الداعمة هي تحسن التسليمات وتحديثات إيجابية من بعض بيوت الخبرة.\n"
                "في المقابل، لا تزال المنافسة تشكل ضغطًا محتملًا على الهوامش.\n"
                "بناءً على ذلك، التوقع الأقرب هو ارتفاع طفيف للسهم.\n"
                "هذا السيناريو يعكس توازنًا بين الزخم الإيجابي والمخاطر التشغيلية."
            )
        return f"Mock response for: {user_query}"

    @observe(name="tool_call", as_type="tool")
    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Registry lookup + loop detection + asyncio.to_thread + error handling."""
        # NOTE: @observe automatically captures tool_name and arguments as 'input'
        # and the return value as 'output'. We only use propagate_attributes here 
        # if we need to pass session_id, user_id, or other context to child spans.

        loop_check = self.loop_detector.check_tool_call(tool_name, json.dumps(arguments))
        if loop_check.is_looping:
            logger.warning(
                "loop_detected",
                tool=tool_name,
                strategy=loop_check.strategy,
                message=loop_check.message,
            )
            result = f"SYSTEM: {loop_check.message} (Detection: {loop_check.strategy})"
            return result

        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            logger.error("tool_not_found", tool=tool_name)
            result = f"Error: Tool '{tool_name}' not found on this agent."
            return result

        try:
            result = str(await asyncio.to_thread(tool.execute, **arguments))
        except ValidationError as e:
            logger.warning("tool_validation_failed", tool=tool_name, error=str(e))
            result = f"Error: Tool arguments validation failed. {e}"
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            result = f"Error: {type(e).__name__}: {e}"

        return result
