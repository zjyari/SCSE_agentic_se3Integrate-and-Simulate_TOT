"""
run_planner.py
==============
Entry point for the Planner Agent.

Responsibility:
  1. Read the validated requirements from artifacts/requirements.json
     (produced by the Analyst Agent in the Requirements Engineering phase).
  2. Call run_planner(requirement) to get a validated navigation plan.
  3. Save the plan to artifacts/plan.json.

plan.json is the artifact that the Developer Agent will consume next.
It contains:
  - strategy       : overall navigation strategy
  - decisions      : list of decisions the robot can take
  - stop_condition : the point at which the robot must stop
"""

import json
import os

from planner_agent import run_planner

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_PATH = os.path.join(HERE, "artifacts", "requirements.json")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
PLAN_PATH = os.path.join(ARTIFACTS_DIR, "plan.json")


def main() -> None:
    if not os.path.exists(REQUIREMENTS_PATH):
        raise FileNotFoundError(
            f"Requirements artifact not found: {REQUIREMENTS_PATH}\n"
            "Run `python run_analyst.py` first to generate it."
        )

    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        requirements = json.load(f)

    # Planner Agent: requirements artifact -> Qwen -> JSON -> validated plan.
    plan = run_planner(requirements)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"Validated navigation plan saved to: {PLAN_PATH}\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
