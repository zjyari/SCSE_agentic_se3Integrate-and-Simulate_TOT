"""
run_analyst.py
==============
Program entry point for the Analyst Agent.

Responsibility:
  1. Read the brief text (brief.txt).
  2. Call run_analyst(brief_text) to get validated software requirements.
  3. Save the final validated requirements as requirements.json inside a
     folder called artifacts/.

Execution flow:
  brief.txt -> run_analyst.py -> run_analyst(brief_text)
           -> system prompt + brief -> call Qwen -> JSON text
           -> validation -> requirements.json
"""

import json
import os

from analyst_agent import run_analyst

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF_PATH = os.path.join(HERE, "brief.txt")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "requirements.json")


def main() -> None:
    with open(BRIEF_PATH, "r", encoding="utf-8") as f:
        brief_text = f.read()

    # Analyst Agent: brief text -> Qwen -> JSON -> validated Python dict.
    requirements = run_analyst(brief_text)

    # Persist the validated requirements.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2, ensure_ascii=False)

    print(f"Validated requirements saved to: {OUTPUT_PATH}\n")
    print(json.dumps(requirements, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
