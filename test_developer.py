"""
test_developer.py
=================
Level 1 - Agent / Pipeline smoke test for the Developer Agent.

Takes the plan artifact (artifacts/plan.json), sends it to the Developer
Agent, lets the agent run as usual (its own validation accepts or rejects the
output), stores the resulting Python code in artifacts/navigation_logic.py and
displays it after the test ends.

Different navigation logic in navigation_logic.py is fine, but the developer
agent MUST create decide_next_move(state) -- the function that other
functions/programs will call.
"""

import ast
import json
import os

from developer_agent import run_developer

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_PATH = os.path.join(HERE, "artifacts", "plan.json")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "navigation_logic.py")


def main() -> None:
    print("=== Test: Developer Agent (Level 1 smoke test) ===\n")

    if not os.path.exists(PLAN_PATH):
        raise FileNotFoundError(
            f"Plan artifact not found: {PLAN_PATH}\n"
            "Run `python test_planner.py` (or `python run_planner.py`) first."
        )

    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # Run the actual agent with its real input (context isolation: only the
    # plan artifact).
    code = run_developer(plan)

    # Store the output in the artifacts folder under navigation_logic.py.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Developer Agent output stored in: {OUTPUT_PATH}\n")
    print("Generated navigation logic:")
    print(code)

    # Level 1 pass conditions: valid Python and it defines decide_next_move.
    tree = ast.parse(code)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "decide_next_move" in defined, (
        "developer agent did not create decide_next_move(state)"
    )
    print("\nDEVELOPER TEST PASSED: navigation_logic.py defines decide_next_move(state).")


if __name__ == "__main__":
    main()
