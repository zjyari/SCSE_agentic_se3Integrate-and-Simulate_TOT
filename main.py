"""
main.py
=======
Orchestrator for the SCSE '26 Integration and Simulation project.

Integrates the separate agents so that they all run one after the other from
this central location. The pipeline:

  1. Load the brief
  2. Run the Analyst agent        -> artifacts/requirements.json
  3. Run the Planner agent        -> artifacts/plan.json
  4. Run the Developer agent      -> artifacts/navigation_logic.py + generated/navigation_logic.py
  5. Run the Test agent           -> generated/test_decide_next_move.py
  6. Run the simulation (runner)  -> artifacts/simulation_report.json
  7. Run the reviewer agent       -> artifacts/review_report.json
  8. Run the IRSIM benchmark      -> artifacts/irsim_benchmark_report.json
  9. Run the benchmark reviewer   -> artifacts/benchmark_review_report.json

  (10. If a revision is needed, generate revised navigation logic, re-test it,
       and re-run the simulation + IR-SIM benchmark.)

Everything is recorded in a log file saved as trace.log.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

from analyst_agent import run_analyst
from benchmark_irsim import main as run_irsim_benchmark
from developer_agent import run_developer, run_developer_revision
from planner_agent import run_planner
from reviewer_agent import run_reviewer
from reviewer_benchmark_agent import run_benchmark_reviewer
from runner import run_simulation
from test_agent import run_test_agent

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
GENERATED_DIR = os.path.join(HERE, "generated")
BRIEF_PATH = os.path.join(HERE, "brief.txt")
TRACE_LOG = os.path.join(HERE, "trace.log")


def log_event(message: str) -> None:
    """Append an event with a timestamp to trace.log and print it."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    with open(TRACE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(filename: str, data) -> str:
    """Save data as JSON under artifacts/ and return the path."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # Reset trace log for a fresh run.
    with open(TRACE_LOG, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pipeline started\n")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # 1. Load the brief -----------------------------------------------------
    with open(BRIEF_PATH, "r", encoding="utf-8") as f:
        brief_text = f.read()
    log_event("Loaded brief from brief.txt")

    # 2. Analyst Agent ------------------------------------------------------
    requirements = run_analyst(brief_text)
    req_path = save_json("requirements.json", requirements)
    log_event(f"Analyst Agent created {req_path}")

    # 3. Planner Agent ------------------------------------------------------
    plan = run_planner(requirements)
    plan_path = save_json("plan.json", plan)
    log_event(f"Planner Agent created {plan_path}")

    # 4. Developer Agent ----------------------------------------------------
    code = run_developer(plan)
    nav_path = save_json("navigation_logic.py", code)
    gen_path = os.path.join(GENERATED_DIR, "navigation_logic.py")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(code)
    log_event(f"Developer Agent created {nav_path} and {gen_path}")

    # 5. Test Agent ---------------------------------------------------------
    test_code = run_test_agent(requirements, plan)
    test_path = os.path.join(GENERATED_DIR, "test_decide_next_move.py")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_code)
    log_event(f"Test Agent created {test_path}")
    # Run the generated tests (requires pytest).
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-q"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120,
        )
        log_event(
            f"Test Agent tests executed (pytest exit={proc.returncode}): "
            f"{proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''}"
        )
    except FileNotFoundError:
        log_event("pytest not installed; generated tests were not executed")

    # 6. Simulation (runner) ------------------------------------------------
    sim_report = run_simulation()
    sim_path = save_json("simulation_report.json", sim_report)
    log_event(
        f"Simulation Runner created {sim_path} "
        f"(passed={sim_report['passed']}/{sim_report['total_scenarios']}, "
        f"success_rate={sim_report['success_rate']})"
    )

    # 7. Reviewer Agent -----------------------------------------------------
    review = run_reviewer(sim_report)
    review_path = save_json("review_report.json", review)
    log_event(f"Reviewer Agent created {review_path} (needs_revision={review['needs_revision']})")

    # 8. IRSIM benchmark ----------------------------------------------------
    try:
        run_irsim_benchmark()
        bench_path = os.path.join(ARTIFACTS_DIR, "irsim_benchmark_report.json")
        benchmark_report = load_json(bench_path)
        log_event(
            f"IRSIM Benchmark created {bench_path} "
            f"(success_rate={benchmark_report['success_rate']}, "
            f"efficiency_rate={benchmark_report['efficiency_rate']})"
        )
    except Exception as exc:  # noqa: BLE001 - keep pipeline alive, log it
        log_event(f"IRSIM Benchmark failed: {exc!r}")
        benchmark_report = {
            "total_worlds": 0, "goals_reached": 0, "success_rate": 0.0,
            "efficient_worlds": 0, "inefficient_success_worlds": 0,
            "failed_worlds": 0, "efficiency_rate": 0.0,
            "average_steps_taken": 0.0, "results": [],
        }

    # 9. Benchmark Reviewer -------------------------------------------------
    bench_review = run_benchmark_reviewer(benchmark_report)
    bench_review_path = save_json("benchmark_review_report.json", bench_review)
    log_event(
        f"Benchmark Reviewer created {bench_review_path} "
        f"(needs_revision={bench_review['needs_revision']})"
    )

    # 10. Revision (if needed) ----------------------------------------------
    if bench_review["needs_revision"] or review["needs_revision"]:
        log_event("Revision required by reviewer -- generating revised navigation logic")
        revised_code = run_developer_revision(
            original_code=code,
            review_notes=review,
            benchmark_report=benchmark_report,
            requirements=requirements,
            plan=plan,
        )
        revised_path = os.path.join(GENERATED_DIR, "navigation_logic_revised.py")
        log_event(f"Developer Agent (revision) created {revised_path}")

        # Re-test the revised logic: swap it in and re-run the simulation.
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(revised_code)
        sim2 = run_simulation()
        sim2_path = save_json("simulation_report_revised.json", sim2)
        log_event(
            f"Revised logic re-tested: {sim2_path} "
            f"(passed={sim2['passed']}/{sim2['total_scenarios']}, "
            f"success_rate={sim2['success_rate']})"
        )

        # Re-run the IR-SIM benchmark with the revised logic.
        try:
            run_irsim_benchmark()
            bench2 = load_json(os.path.join(ARTIFACTS_DIR, "irsim_benchmark_report.json"))
            log_event(
                f"IRSIM Benchmark re-run with revised logic "
                f"(success_rate={bench2['success_rate']}, "
                f"efficiency_rate={bench2['efficiency_rate']})"
            )
        except Exception as exc:  # noqa: BLE001
            log_event(f"IRSIM Benchmark re-run failed: {exc!r}")
    else:
        log_event("No revision required -- navigation logic accepted")

    log_event("Pipeline finished")


if __name__ == "__main__":
    main()
