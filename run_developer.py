"""
run_developer.py
================
Entry point for the Developer Agent.

Responsibility:
  1. Read the validated navigation plan from artifacts/plan.json
     (produced by the Planner Agent).
  2. Call run_developer(plan) to get validated Python code.
  3. Save the generated code as navigation_logic.py.

The logic here mirrors run_analyst.py and run_planner.py.
"""

import json
import os

from developer_agent import run_developer

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_PATH = os.path.join(HERE, "artifacts", "plan.json")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "navigation_logic.py")


def main() -> None:
    if not os.path.exists(PLAN_PATH):
        raise FileNotFoundError(
            f"Plan artifact not found: {PLAN_PATH}\n"
            "Run `python run_planner.py` first to generate it."
        )

    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # Developer Agent: plan -> Qwen -> Python code -> validated code string.
    code = run_developer(plan)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Generated navigation logic saved to: {OUTPUT_PATH}\n")
    print(code)


if __name__ == "__main__":
    main()
