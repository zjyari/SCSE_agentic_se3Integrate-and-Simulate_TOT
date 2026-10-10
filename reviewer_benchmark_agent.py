"""
reviewer_benchmark_agent.py
===========================
Benchmark Reviewer Agent for the SCSE '26 Integration and Simulation project.

Reviews the IR-SIM benchmark report, measures quality requirements such as
safety, goal completion and efficiency, and produces a review report that
tells the result of the review and recommends further actions (e.g. whether
revision of the code is needed). It also specifies precisely what issue it
found in the report and how that can be improved/fixed.

Output schema (strict JSON):
{
  "issue": "string",
  "likely_cause": "string",
  "recommendation": "string",
  "safety_passed": true|false,
  "goal_completion_passed": true|false,
  "efficiency_passed": true|false,
  "needs_revision": true|false
}

The pass/fail booleans are ALWAYS derived deterministically from the measured
benchmark fields (build_deterministic_review); the LLM only phrases the issue,
likely cause and recommendation.
"""

import json
import os
import re
from typing import Any, Dict

import dashscope

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

REQUIRED_BENCH_REVIEW_KEYS = {
    "issue",
    "likely_cause",
    "recommendation",
    "safety_passed",
    "goal_completion_passed",
    "efficiency_passed",
    "needs_revision",
}
BOOLEAN_KEYS = {"safety_passed", "goal_completion_passed", "efficiency_passed", "needs_revision"}

SYSTEM_PROMPT = """You are a Benchmark Reviewer Agent in a multi-agent software system.

ROLE:
An automated IR-SIM benchmark has measured the robot's navigation logic
(decide_next_move) across several simulated environments. You review the
benchmark report and recommend further actions.

TASK:
Analyze the IR-SIM benchmark report JSON and the deterministic review provided
by the user, and produce a SINGLE JSON object with EXACTLY these seven keys:
  - "issue": a string describing the precise issue found in the report, or
    "No issues found" if everything passed.
  - "likely_cause": a string describing the most likely cause (or "" if no issue).
  - "recommendation": a string recommending how to improve/fix (or "" if no issue).
  - "safety_passed": boolean - whether safety passed.
  - "goal_completion_passed": boolean - whether goal completion passed.
  - "efficiency_passed": boolean - whether efficiency passed.
  - "needs_revision": boolean - whether the code needs revision.

IMPORTANT: The deterministic review contains the ground-truth pass/fail
booleans derived from the measured benchmark numbers. Copy those booleans
EXACTLY as given. Only the issue / likely_cause / recommendation wording may
be phrased by you, and it must be consistent with those booleans.

CONSTRAINTS:
1. Base your analysis ONLY on the benchmark report and the deterministic review.
2. Output ONLY the JSON object. No prose, no explanation, no markdown code
   fences (no ```json ... ```), no trailing comments.

OUTPUT STRUCTURE (copy this shape exactly):
{
  "issue": "...",
  "likely_cause": "...",
  "recommendation": "...",
  "safety_passed": true,
  "goal_completion_passed": true,
  "efficiency_passed": true,
  "needs_revision": false
}
"""


def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object out of a model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return text[start : end + 1]


def validate_benchmark_review(data: Any) -> Dict[str, Any]:
    """Validate the Benchmark Reviewer's output.

    Checks: dict, exactly the seven required keys, the four boolean keys are
    booleans, and the remaining keys are strings. Raises ValueError otherwise.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict, got {type(data).__name__}")

    actual_keys = set(data.keys())
    missing = REQUIRED_BENCH_REVIEW_KEYS - actual_keys
    extra = actual_keys - REQUIRED_BENCH_REVIEW_KEYS
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected extra keys introduced by Qwen: {sorted(extra)}")

    for key in BOOLEAN_KEYS:
        if not isinstance(data[key], bool):
            raise ValueError(f"'{key}' must be a boolean")

    for key in ("issue", "likely_cause", "recommendation"):
        if not isinstance(data[key], str):
            raise ValueError(f"'{key}' must be a string")

    return data


def build_deterministic_review(benchmark_report: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic review from the measured benchmark fields.

    Safety        : any world with collision_or_done true while reached_goal is
                    false -> safety_passed False.
    Goal completion: success_rate < 1.0 -> goal_completion_passed False.
    Efficiency    : inefficient_success_worlds > 0 or efficiency_rate < 1.0
                    -> efficiency_passed False.
    needs_revision: True if any of the three did not pass.
    If the report lacks required fields, default values are used.
    """
    total = benchmark_report.get("total_worlds", 0)
    success_rate = benchmark_report.get("success_rate", 0.0)
    efficiency_rate = benchmark_report.get("efficiency_rate", 0.0)
    inefficient_success_worlds = benchmark_report.get("inefficient_success_worlds", 0)

    results = benchmark_report.get("results", [])
    unsafe_world = False
    for entry in results:
        if entry.get("collision_or_done") and not entry.get("reached_goal"):
            unsafe_world = True
            break

    safety_passed = not unsafe_world
    goal_completion_passed = success_rate >= 1.0
    efficiency_passed = (
        inefficient_success_worlds == 0 and efficiency_rate >= 1.0
    )
    needs_revision = not (safety_passed and goal_completion_passed and efficiency_passed)

    issues = []
    if not safety_passed:
        issues.append("Unsafe benchmark behavior detected")
    if not goal_completion_passed:
        issues.append("Goal completion failure")
    if not efficiency_passed:
        issues.append("Successful but inefficient navigation")

    if issues:
        issue_text = "; ".join(issues)
        likely_cause = (
            "The navigation logic does not satisfy one or more measured "
            "quality criteria in the IR-SIM environments."
        )
        recommendation = (
            "Revise decide_next_move: ensure it never drives into blocked "
            "directions, always reaches the goal, and takes the most direct "
            "safe path."
        )
    else:
        issue_text = "No issues found"
        likely_cause = ""
        recommendation = "No revision required; keep the current navigation logic."

    return {
        "issue": issue_text,
        "likely_cause": likely_cause,
        "recommendation": recommendation,
        "safety_passed": safety_passed,
        "goal_completion_passed": goal_completion_passed,
        "efficiency_passed": efficiency_passed,
        "needs_revision": needs_revision,
    }


def _call_qwen(user_content: str) -> str:
    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        result_format="message",
    )
    return response.output.choices[0].message.content


def run_benchmark_reviewer(benchmark_report: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    """Run the Benchmark Reviewer Agent.

    Takes the benchmark report, builds a deterministic review, sends both to
    the Reviewer Agent, and returns the validated output with the pass/fail
    booleans taken from the deterministic (measured) review.
    """
    if not dashscope.api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Export it before running, e.g.\n"
            '  PowerShell:  $env:DASHSCOPE_API_KEY = "sk-..."\n'
            "  bash/zsh:    export DASHSCOPE_API_KEY=sk-..."
        )

    deterministic = build_deterministic_review(benchmark_report)
    user_content = (
        "IR-SIM benchmark report:\n"
        f"{json.dumps(benchmark_report, indent=2, ensure_ascii=False)}\n\n"
        "Deterministic review (ground-truth pass/fail booleans):\n"
        f"{json.dumps(deterministic, indent=2, ensure_ascii=False)}"
    )

    extra_instruction = ""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        raw = _call_qwen(user_content + extra_instruction)
        try:
            data = json.loads(_extract_json(raw))
            validated = validate_benchmark_review(data)
        except json.JSONDecodeError as exc:
            last_error = ValueError(f"Qwen returned invalid JSON: {exc}\nRaw: {raw}")
            extra_instruction = (
                f"\n\n---\nYour previous output was not valid: {last_error}\n"
                f"Return ONLY a single valid JSON object with EXACTLY the keys "
                f"{sorted(REQUIRED_BENCH_REVIEW_KEYS)} and correct types."
            )
            continue
        except (ValueError, TypeError) as exc:
            last_error = exc
            extra_instruction = (
                f"\n\n---\nYour previous output was not valid: {last_error}\n"
                f"Return ONLY a single valid JSON object with EXACTLY the keys "
                f"{sorted(REQUIRED_BENCH_REVIEW_KEYS)} and correct types."
            )
            continue

        # Ground-truth booleans always come from the measured benchmark review.
        for key in BOOLEAN_KEYS:
            validated[key] = deterministic[key]
        return validated

    raise RuntimeError(
        f"Benchmark Reviewer failed after {max_retries + 1} attempts: {last_error}"
    )
