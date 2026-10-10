"""
benchmark_irsim.py
==================
Automated IR-SIM benchmarking for the SCSE '26 Integration and Simulation
project.

Loads all the IR-SIM world files one by one by calling
test_irsim_with_agent.py, captures each world's report, and produces a summary
benchmark report with metrics:

  - Success rate : goals reached / total environments
  - Efficiency rate : efficient runs / total environments
  - Average steps : total steps / total environments

The summary report is saved to artifacts/irsim_benchmark_report.json.
"""

import ast
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
IRSIM_SCRIPT = os.path.join(HERE, "test_irsim_with_agent.py")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "irsim_benchmark_report.json")

# The four simulated environments of different difficulty level.
WORLD_FILES = [
    os.path.join(HERE, "irsim_worlds", "world_easy.yaml"),
    os.path.join(HERE, "irsim_worlds", "world_medium.yaml"),
    os.path.join(HERE, "irsim_worlds", "world_hard.yaml"),
    os.path.join(HERE, "irsim_worlds", "world_custom.yaml"),
]


def run_irsim_benchmark(world_file: str) -> dict:
    """Run the IR-SIM test for one world file.

    Calls test_irsim_with_agent.py with the world file as an argument, captures
    the output, and extracts the report dict from the printed "IR-SIM report:"
    section. The matplotlib Agg backend is forced so the automated benchmark
    runs headless (no pop-up windows).
    """
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # headless rendering for automated runs

    proc = subprocess.run(
        [sys.executable, IRSIM_SCRIPT, world_file],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=600,
    )

    output = proc.stdout + proc.stderr
    match = re.search(r"IR-SIM report:\s*(\{.*\})", output, re.DOTALL)
    if not match:
        raise RuntimeError(
            f"Could not find IR-SIM report in output for {world_file}.\n{output[-2000:]}"
        )

    try:
        report = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"Could not parse IR-SIM report for {world_file}: {exc}") from exc

    return report


def main() -> None:
    results = []
    reached_goal_count = 0
    efficient_count = 0
    inefficient_success_count = 0
    failed_count = 0
    total_steps = 0

    for world_file in WORLD_FILES:
        print(f"\n=== Benchmarking world: {os.path.basename(world_file)} ===")
        report = run_irsim_benchmark(world_file)

        reached = bool(report.get("reached_goal"))
        efficiency_status = report.get("efficiency_status", "failed")
        steps = int(report.get("steps_taken", 0))
        collision = bool(report.get("collision_or_done"))

        results.append(
            {
                "world": os.path.basename(world_file),
                "reached_goal": reached,
                "collision_or_done": collision,
                "steps_taken": steps,
                "efficiency_status": efficiency_status,
                "efficiency_passed": bool(report.get("efficiency_passed", False)),
            }
        )

        total_steps += steps
        if reached:
            reached_goal_count += 1
            if efficiency_status == "efficient":
                efficient_count += 1
            else:
                inefficient_success_count += 1
        else:
            failed_count += 1

        print(f"  reached_goal={reached}, status={efficiency_status}, steps={steps}")

    total = len(WORLD_FILES)
    average_steps = round(total_steps / total, 2) if total else 0.0

    benchmark_report = {
        "total_worlds": total,
        "goals_reached": reached_goal_count,
        "success_rate": round(reached_goal_count / total, 4) if total else 0.0,
        "efficient_worlds": efficient_count,
        "inefficient_success_worlds": inefficient_success_count,
        "failed_worlds": failed_count,
        "efficiency_rate": round(efficient_count / total, 4) if total else 0.0,
        "average_steps_taken": average_steps,
        "results": results,
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    print("\n=== IR-SIM benchmark report ===")
    print(f"Total worlds            : {benchmark_report['total_worlds']}")
    print(f"Goals reached           : {benchmark_report['goals_reached']}")
    print(f"Success rate            : {benchmark_report['success_rate']:.2%}")
    print(f"Efficient worlds        : {benchmark_report['efficient_worlds']}")
    print(f"Inefficient but success : {benchmark_report['inefficient_success_worlds']}")
    print(f"Failed worlds           : {benchmark_report['failed_worlds']}")
    print(f"Efficiency rate         : {benchmark_report['efficiency_rate']:.2%}")
    print(f"Average steps taken     : {benchmark_report['average_steps_taken']}")
    print(f"\nSaved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
