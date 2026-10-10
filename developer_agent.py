"""
developer_agent.py
==================
Developer Agent for the SCSE '26 Plan and Develop / Integration and
Simulation project.

The Planner Agent has produced a navigation plan (plan.json). The Developer
Agent's job is to turn that plan into actual, working Python code.

It asks Qwen to write a Python module implementing the navigation logic, then
validates that the output is proper Python code (syntactically valid and
defining the main function decide_next_move(state)) using validate_code().

The generated code is saved as navigation_logic.py by run_developer.py.

In the Integration and Simulation phase, the agent also supports revision:
run_developer_revision() receives the original navigation code, the review
notes, the benchmark report, the requirements and the plan, asks the LLM to
generate a new navigation logic, and saves it as a new file.
"""

import ast
import json
import os
import re
from typing import Any, Dict

import dashscope

# API key from environment -- never hard-code secrets.
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# --- Prompt: role, task, constraints, output structure ---------------------
SYSTEM_PROMPT = """You are a Developer Agent in a multi-agent software system.

ROLE:
The Planner Agent has produced a navigation plan. Your job is to implement that
plan as working, self-contained Python code.

TASK:
Write a Python module that implements the robot navigation logic described by
the plan.

The module MUST define a main function with this exact signature:

    def decide_next_move(state):

where `state` is the input the robot receives from its sensor: a dict with
these six boolean keys:
  - "goal_ahead", "goal_on_left", "goal_on_right" (where the goal is)
  - "front_blocked", "left_blocked", "right_blocked" (which paths are blocked)

The function must return exactly one of the allowed actions:
"FORWARD", "LEFT", "RIGHT", or "STOP".

You may keep any other helper functions inside the module, but OTHER agents and
programs will call ONLY decide_next_move(state) -- no other function.

CONSTRAINTS AND RULES:
1. Output ONLY valid Python code. No prose, no explanation, no markdown code
   fences (no ```python ... ```), no trailing comments outside the code.
2. The code must be syntactically valid Python 3.
3. It must define the main function decide_next_move(state); helper functions
   are allowed.
4. Only use the actions FORWARD, LEFT, RIGHT, STOP -- never invent new actions.
5. The logic must respect the plan: prefer moving toward the goal when safe,
   never move into a blocked direction, and stop when no safe direction exists.
6. Keep it self-contained: standard library only, no third-party imports.
7. Include a short docstring on decide_next_move describing the state dict.

Output the raw Python source code only.
"""


def _extract_code(text: str) -> str:
    """Extract Python source from a model reply.

    Handles the common cases where Qwen:
      * wraps the code in ```python ... ``` fences,
      * adds prose before or after the code block,
      * returns bare code with no fences.
    """
    text = text.strip()

    # 1. Try to capture the content of the first fenced code block.
    fence_match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        code = fence_match.group(1)
    else:
        # 2. No fences: find the first line that looks like Python and keep
        #    everything from there onward.
        lines = text.splitlines()
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("def ", "class ", "import ", "from ", "#")):
                start_idx = i
                break
        code = "\n".join(lines[start_idx:])

    return code.strip() + "\n"


def validate_code(code: Any) -> str:
    """Validate that the Developer Agent's output is proper Python code.

    Checks:
      * code is a non-empty string
      * it parses as valid Python (ast.parse)
      * it defines the main function decide_next_move(state)
        (helper functions are allowed in addition)
    """
    if not isinstance(code, str) or not code.strip():
        raise TypeError("Developer agent returned empty or non-string code")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Generated code is not valid Python: {exc}") from exc

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "decide_next_move" not in defined:
        raise ValueError(
            "Generated code must define the main function decide_next_move(state)"
        )

    return code


def _call_qwen(plan_json: str, extra_instruction: str = "") -> str:
    """Send the system prompt + plan to Qwen, return raw text."""
    user_content = f"Navigation plan:\n\n{plan_json}"
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


def run_developer(plan: Dict[str, Any], max_retries: int = 2) -> str:
    """Developer Agent entry point.

    Takes the validated plan (from the Planner Agent), sends it to Qwen,
    extracts the Python code from the reply, validates it with validate_code(),
    and returns the code as a string. Retries with a correction instruction if
    the output does not validate.
    """
    if not dashscope.api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Export it before running, e.g.\n"
            '  PowerShell:  $env:DASHSCOPE_API_KEY = "sk-..."\n'
            "  bash/zsh:    export DASHSCOPE_API_KEY=sk-..."
        )

    plan_json = json.dumps(plan, indent=2, ensure_ascii=False)
    extra_instruction = ""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        raw = _call_qwen(plan_json, extra_instruction)
        try:
            code = _extract_code(raw)
            return validate_code(code)
        except (ValueError, TypeError) as exc:
            last_error = exc

        extra_instruction = (
            f"Your previous output was not valid Python code: {last_error}\n"
            f"Return ONLY raw, syntactically valid Python code that defines the "
            f"main function decide_next_move(state) (state is a dict with the "
            f"six boolean keys goal_ahead, goal_on_left, goal_on_right, "
            f"front_blocked, left_blocked, right_blocked) and returns one of "
            f"FORWARD, LEFT, RIGHT, STOP. No prose, no markdown fences."
        )

    raise RuntimeError(
        f"Developer agent failed after {max_retries + 1} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Integration and Simulation phase: developer revision
# ---------------------------------------------------------------------------
REVISION_SYSTEM_PROMPT = """You are a Developer Agent revising navigation logic in a multi-agent software system.

ROLE:
A reviewer has reviewed the simulation/benchmark of the current navigation
logic (decide_next_move) and found problems. Your job is to produce a REVISED
version of the navigation logic that fixes those problems, based on the
original code, the review notes, the benchmark report, the requirements and
the plan.

INPUT:
You receive:
  - the original navigation code,
  - the review notes (issues found by the reviewer),
  - the benchmark report,
  - the requirements artifact,
  - the plan artifact.

TASK:
Write a REVISED Python module implementing the robot navigation logic. Keep
the same public interface:

    def decide_next_move(state):

where `state` is a dict with six boolean keys:
  - "goal_ahead", "goal_on_left", "goal_on_right" (where the goal is)
  - "front_blocked", "left_blocked", "right_blocked" (which paths are blocked)

The function must return exactly one of: "FORWARD", "LEFT", "RIGHT", "STOP".

CONSTRAINTS AND RULES:
1. Fix the issues described in the review notes and address the metrics in the
   benchmark report (safety: never move into a blocked direction; goal
   completion: always reach the goal when possible; efficiency: take the most
   direct safe path).
2. The benchmark failure was caused by the robot STOPPING as soon as its goal
   direction was blocked. To reach the goal in a real environment the robot
   MUST be able to navigate AROUND obstacles. Therefore implement obstacle
   avoidance with this decision order:
     1) if goal ahead and front clear          -> FORWARD
     2) if goal on left and left clear         -> LEFT
     3) if goal on right and right clear       -> RIGHT
     4) otherwise, if any direction is clear, pick a clear direction to
        manoeuvre around the obstacle (prefer FORWARD, then LEFT, then RIGHT)
     5) only if ALL directions are blocked     -> STOP
3. Keep the function name and signature decide_next_move(state).
4. Only use the actions FORWARD, LEFT, RIGHT, STOP -- never invent new actions.
5. Output ONLY valid Python 3 code. No prose, no markdown fences
   (no ```python ... ```), no trailing comments outside the code.
6. Standard library only, no third-party imports.

Output the raw Python source code only.
"""


def _call_revision_qwen(original_code: str, review_notes: Dict, benchmark_report: Dict,
                        requirements: Dict, plan: Dict, extra_instruction: str = "") -> str:
    """Send the revision prompt with all context to Qwen, return raw text."""
    user_content = (
        "Original navigation code:\n"
        f"{original_code}\n\n"
        "Review notes:\n"
        f"{json.dumps(review_notes, indent=2, ensure_ascii=False)}\n\n"
        "Benchmark report:\n"
        f"{json.dumps(benchmark_report, indent=2, ensure_ascii=False)}\n\n"
        "Requirements artifact:\n"
        f"{json.dumps(requirements, indent=2, ensure_ascii=False)}\n\n"
        "Plan artifact:\n"
        f"{json.dumps(plan, indent=2, ensure_ascii=False)}"
    )
    if extra_instruction:
        user_content += f"\n\n---\n{extra_instruction}"

    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        result_format="message",
    )
    return response.output.choices[0].message.content


def run_developer_revision(original_code: str, review_notes: Dict[str, Any],
                           benchmark_report: Dict[str, Any],
                           requirements: Dict[str, Any],
                           plan: Dict[str, Any],
                           output_path: str | None = None,
                           max_retries: int = 2) -> str:
    """Generate a revised navigation logic.

    Receives the original navigation code, the review notes, the benchmark
    report, the requirements and the plan; asks the LLM to generate a new
    navigation logic, validates it, saves it as a new file, and returns the
    new code.

    Args:
        original_code: the current navigation logic source (string).
        review_notes: review output dict from the reviewer agent(s).
        benchmark_report: the IR-SIM benchmark report dict.
        requirements: the requirements artifact dict.
        plan: the plan artifact dict.
        output_path: where to save the revised code. Defaults to
            <project>/generated/navigation_logic_revised.py.
        max_retries: how many correction attempts after a failed validation.

    Returns:
        The revised navigation logic source code (string).
    """
    if not dashscope.api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Export it before running, e.g.\n"
            '  PowerShell:  $env:DASHSCOPE_API_KEY = "sk-..."\n'
            "  bash/zsh:    export DASHSCOPE_API_KEY=sk-..."
        )

    if output_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(here, "generated", "navigation_logic_revised.py")

    extra_instruction = ""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        raw = _call_revision_qwen(
            original_code, review_notes, benchmark_report, requirements, plan,
            extra_instruction,
        )
        try:
            code = _extract_code(raw)
            code = validate_code(code)
        except (ValueError, TypeError) as exc:
            last_error = exc
            extra_instruction = (
                f"Your previous output was not valid Python code: {last_error}\n"
                f"Return ONLY raw, syntactically valid Python code that defines "
                f"decide_next_move(state) and returns one of FORWARD, LEFT, "
                f"RIGHT, STOP. No prose, no markdown fences."
            )
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return code

    raise RuntimeError(
        f"Developer revision failed after {max_retries + 1} attempts: {last_error}"
    )
