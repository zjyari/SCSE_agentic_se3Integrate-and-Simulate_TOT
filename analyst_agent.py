"""
analyst_agent.py
================
Analyst Agent for the SCSE '26 Requirements Engineering project.

Responsibility (like a human requirements engineer):
    Read a customer's natural-language brief about a mobile robot and ask Qwen
    to extract explicit, machine-readable software requirements as strict JSON.

The rest of the system consumes this agent's output, so the JSON MUST be valid
and use EXACTLY the structure required by the assignment:

    {
      "goal": "string",
      "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
      "safe_stop": true,
      "avoid.obstacles": true
    }

NOTE: the fourth key is literally "avoid.obstacles" (with a dot), exactly as
written in the project instructions PDF. The final navigation logic of the whole
system may only ever be: FORWARD, LEFT, RIGHT, STOP -- no other action.

The JSON itself MUST come from Qwen; this module only orchestrates the call,
parsing and validation (it never hard-codes the requirements).
"""

import json
import os
import re
from typing import Any, Dict

import dashscope

# Read the API key from the environment -- never hard-code secrets.
# Set it beforehand, e.g. (PowerShell):  $env:DASHSCOPE_API_KEY = "sk-..."
# or (bash/zsh):                      export DASHSCOPE_API_KEY="sk-..."
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# --- Fixed contract that the rest of the system relies on -------------------
# The fourth key is literally "avoid.obstacles" (with a dot), exactly as shown
# in the project instructions PDF.
REQUIRED_KEYS = {"goal", "allowed_actions", "safe_stop", "avoid.obstacles"}
PERMITTED_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}

# --- Prompt: role, task, constraints, output structure ---------------------
SYSTEM_PROMPT = """You are a professional requirements engineer (an Analyst Agent).

ROLE:
You behave like a human requirements engineer. A customer hands you a short,
natural-language brief about a mobile robot; your job is to read it and extract
the specific, explicit software requirements that an engineer can build and test.

TASK:
Read the robot navigation brief given by the user and produce a SINGLE,
strictly-formatted JSON object that captures the navigation requirements.

CONSTRAINTS AND RULES:
1. The JSON object must contain EXACTLY these four keys and no others:
   - "goal": a short string describing the navigation objective understood from
     the brief.
   - "allowed_actions": a list. It must be EXACTLY ["FORWARD", "LEFT", "RIGHT",
     "STOP"]. No additional action may ever be introduced.
   - "safe_stop": a boolean. true if the robot must stop safely when there is
     no safe direction available.
   - "avoid.obstacles": a boolean. true if the robot must avoid anything that
     blocks its path (it must never move into a blocked direction).
2. Do not invent requirements that are not implied by the brief.
3. Output ONLY the JSON object. No prose, no explanation, no markdown code
   fences (no ```json ... ```), no trailing comments.

OUTPUT STRUCTURE (copy this shape exactly):
{
  "goal": "...",
  "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
  "safe_stop": true,
  "avoid.obstacles": true
}
"""


def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object out of a model reply.

    Qwen occasionally wraps the JSON in ```json fences or adds a little
    surrounding prose; this normalises that back to a raw JSON string.
    """
    text = text.strip()

    # Strip a leading ```json / ``` fence if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return text[start : end + 1]


def validate_requirements(data: Any) -> Dict[str, Any]:
    """Validate the Python object that came back from Qwen.

    Checks (per the assignment):
      * the data is a dictionary
      * all required keys exist, and no new key was introduced
      * "goal" is a string
      * "allowed_actions" is a list, containing only the four permitted actions
      * "safe_stop" and "avoid.obstacles" are booleans

    Raises TypeError / ValueError with a descriptive message on failure.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    actual_keys = set(data.keys())
    missing = REQUIRED_KEYS - actual_keys
    extra = actual_keys - REQUIRED_KEYS
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected extra keys introduced by Qwen: {sorted(extra)}")

    if not isinstance(data["goal"], str) or not data["goal"].strip():
        raise TypeError("'goal' must be a non-empty string")

    if not isinstance(data["allowed_actions"], list):
        raise TypeError("'allowed_actions' must be a list")

    invalid_actions = [a for a in data["allowed_actions"] if a not in PERMITTED_ACTIONS]
    if invalid_actions:
        raise ValueError(
            f"Invalid action(s) {invalid_actions}. "
            f"Only {sorted(PERMITTED_ACTIONS)} are allowed."
        )

    if not isinstance(data["safe_stop"], bool):
        raise TypeError("'safe_stop' must be a boolean")

    if not isinstance(data["avoid.obstacles"], bool):
        raise TypeError("'avoid.obstacles' must be a boolean")

    return data


def _call_qwen(brief_text: str, extra_instruction: str = "") -> str:
    """Send the system prompt + brief to Qwen and return the raw text reply."""
    user_content = f"Robot navigation brief:\n\n{brief_text}"
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


def run_analyst(brief_text: str, max_retries: int = 2) -> Dict[str, Any]:
    """Analyst Agent entry point.

    Accepts the brief text as a parameter (so it can be called by another
    function), sends suitable prompts to Qwen, accepts its response, converts
    the JSON text into Python data, validates it, and returns the validated
    requirements.

    The final JSON comes from Qwen; this function only orchestrates the call,
    parsing and validation. It retries a couple of times with a correction
    instruction if Qwen returns something that does not validate.
    """
    if not dashscope.api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Export it before running, e.g.\n"
            '  PowerShell:  $env:DASHSCOPE_API_KEY = "sk-..."\n'
            "  bash/zsh:    export DASHSCOPE_API_KEY=sk-..."
        )

    extra_instruction = ""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        raw = _call_qwen(brief_text, extra_instruction)
        try:
            json_str = _extract_json(raw)
            data = json.loads(json_str)
            return validate_requirements(data)
        except json.JSONDecodeError as exc:
            last_error = ValueError(f"Qwen returned invalid JSON: {exc}\nRaw: {raw}")
        except (ValueError, TypeError) as exc:
            last_error = exc

        # Ask Qwen to correct its output on the next attempt.
        extra_instruction = (
            f"Your previous output was not valid: {last_error}\n"
            f"Return ONLY a single valid JSON object with EXACTLY the four keys "
            f"{sorted(REQUIRED_KEYS)} and correct types, no extra text."
        )

    raise RuntimeError(
        f"Analyst agent failed after {max_retries + 1} attempts: {last_error}"
    )
