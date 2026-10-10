"""
brief_to_req.py
===============
First step of the Requirements Engineering phase.

Ask Qwen to read the human-language brief (brief.txt) and produce
human-readable software requirements as plain text.

Run this several times and examine the output:
  * Is the output consistent across runs?
  * Is the output acceptable?
  * Could it be used directly by another program?

The human-readable requirements are saved to robot_requirements.txt.
"""

import os

import dashscope

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

SYSTEM_PROMPT = """You are a requirements engineer.

Read the robot navigation brief provided by the user and turn it into a clear,
human-readable list of software requirements.

Output rules:
  * Plain text only -- a bulleted or numbered list.
  * Do NOT output JSON.
  * Do not invent requirements that are not implied by the brief.
  * Be precise and unambiguous.
"""


def brief_to_requirements_text(brief_text: str) -> str:
    """Ask Qwen to convert the brief into human-readable requirements text."""
    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": brief_text},
        ],
        result_format="message",
    )
    return response.output.choices[0].message.content


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(here, "brief.txt"), "r", encoding="utf-8") as f:
        brief = f.read()

    req_text = brief_to_requirements_text(brief)

    out_path = os.path.join(here, "robot_requirements.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(req_text)

    print(f"Saved human-readable requirements to: {out_path}\n")
    print(req_text)
