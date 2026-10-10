"""
planner_agent.py
================
Planner Agent for the SCSE '26 Plan and Develop project.

Analyst Agent  -> decides WHAT the software must achieve (requirements).
Planner Agent  -> decides HOW the software should behave (navigation plan).

Context isolation: this agent receives ONLY the validated requirements artifact
from the Analyst Agent -- not the entire Analyst conversation.

It asks Qwen to turn the requirements into a strict JSON plan, then validates
the plan with validate_plan().

The plan.json artifact must contain:
  - strategy       : overall navigation strategy (string)
  - decisions      : list of decisions the robot can take (list of strings)
  - stop_condition : the point at which the robot must stop (string)
"""

import json
import os
import re
from typing import Any, Dict

import dashscope

# API key from environment -- never hard-code secrets.
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# --- Fixed plan contract ----------------------------------------------------
REQUIRED_PLAN_KEYS = {"strategy", "decisions", "stop_condition"}

# --- Prompt: role, task, constraints, output structure ---------------------
SYSTEM_PROMPT = """You are a Planner Agent in a multi-agent software system.

ROLE:
The Analyst Agent has already decided WHAT the software must achieve (the
requirements). Your job is to decide HOW the software should behave: turn the
requirements into a concrete navigation plan that a Developer Agent can later
implement in code.

INPUT:
You will receive a validated requirements JSON artifact with these keys:
  - goal             : the navigation objective
  - allowed_actions  : the ONLY actions the robot may take
  - safe_stop        : whether the robot must stop safely when it cannot go anywhere
  - avoid.obstacles  : whether the robot must avoid anything that blocks its path

TASK:
Produce a SINGLE JSON object describing the navigation plan.

CONSTRAINTS AND RULES:
1. The JSON must contain EXACTLY these three keys and no others:
   - "strategy": a short string describing the overall navigation strategy.
   - "decisions": a non-empty list of strings. Each string describes one
     decision the robot can take during navigation (for example:
     "if the goal is ahead and the front path is clear, move FORWARD").
   - "stop_condition": a string describing the precise point at which the
     robot must stop.
2. Base the plan ONLY on the provided requirements. Do not invent new actions
   beyond FORWARD, LEFT, RIGHT, STOP.
3. The decisions must be ordered logically and must never tell the robot to
   move into a blocked direction.
4. Output ONLY the JSON object. No prose, no explanation, no markdown code
   fences (no ```json ... ```), no trailing comments.

OUTPUT STRUCTURE (copy this shape exactly):
{
  "strategy": "...",
  "decisions": ["...", "..."],
  "stop_condition": "..."
}
"""


def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object out of a model reply.

    Normalises the common case where Qwen wraps JSON in ```json fences or
    adds a little surrounding prose.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return text[start : end + 1]


def validate_plan(data: Any) -> Dict[str, Any]:
    """Validate the plan returned by Qwen.

    Checks:
      * data is a dict
      * exactly the keys strategy, decisions, stop_condition (no missing, no extra)
      * strategy is a non-empty string
      * decisions is a non-empty list of non-empty strings
      * stop_condition is a non-empty string
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    actual_keys = set(data.keys())
    missing = REQUIRED_PLAN_KEYS - actual_keys
    extra = actual_keys - REQUIRED_PLAN_KEYS
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected extra keys introduced by Qwen: {sorted(extra)}")

    if not isinstance(data["strategy"], str) or not data["strategy"].strip():
        raise TypeError("'strategy' must be a non-empty string")

    if not isinstance(data["decisions"], list) or len(data["decisions"]) == 0:
        raise TypeError("'decisions' must be a non-empty list")
    for idx, decision in enumerate(data["decisions"]):
        if not isinstance(decision, str) or not decision.strip():
            raise TypeError(f"decisions[{idx}] must be a non-empty string")

    if not isinstance(data["stop_condition"], str) or not data["stop_condition"].strip():
        raise TypeError("'stop_condition' must be a non-empty string")

    return data


def _call_qwen(requirement_json: str, extra_instruction: str = "") -> str:
    """Send the system prompt + requirements artifact to Qwen, return raw text."""
    user_content = f"Requirements artifact:\n\n{requirement_json}"
    if extra_instruction:
        user_content += f"\n\n---\n{extra_instruction}"

    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        result_format="message",
    )
    return response.output.choices[0].message.content


def run_planner(requirement: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    """Planner Agent entry point.

    Takes the validated requirements artifact (from the Analyst Agent), sends
    it to Qwen, parses the JSON reply, validates it with validate_plan(), and
    returns the validated plan. Retries with a correction instruction if the
    output does not validate.
    """
    if not dashscope.api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Export it before running, e.g.\n"
            '  PowerShell:  $env:DASHSCOPE_API_KEY = "sk-..."\n'
            "  bash/zsh:    export DASHSCOPE_API_KEY=sk-..."
        )

    requirement_json = json.dumps(requirement, indent=2, ensure_ascii=False)
    extra_instruction = ""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        raw = _call_qwen(requirement_json, extra_instruction)
        try:
            json_str = _extract_json(raw)
            data = json.loads(json_str)
            return validate_plan(data)
        except json.JSONDecodeError as exc:
            last_error = ValueError(f"Qwen returned invalid JSON: {exc}\nRaw: {raw}")
        except (ValueError, TypeError) as exc:
            last_error = exc

        extra_instruction = (
            f"Your previous output was not valid: {last_error}\n"
            f"Return ONLY a single valid JSON object with EXACTLY the three keys "
            f"{sorted(REQUIRED_PLAN_KEYS)} and correct types, no extra text."
        )

    raise RuntimeError(
        f"Planner agent failed after {max_retries + 1} attempts: {last_error}"
    )
