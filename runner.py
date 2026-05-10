import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from browser_automation_agent import AutomationStep, BrowserAutomationAgent
from goal_validator import GoalCriterion, TaskGoal
from logging_config import setup_logging, get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    task_id: str
    url: str
    goal: str
    steps: list[AutomationStep]
    goal_criteria: Optional[list[GoalCriterion]] = None
    category: Optional[str] = None
    credentials: Optional[dict] = None


@dataclass
class TaskResult:
    task_id: str
    url: str
    goal: str
    success: bool
    duration_ms: float
    plan_status: Optional[str]
    steps_total: int
    steps_ok: int
    goal_progress_probability: Optional[float]
    extracted_answer: Optional[str] = None
    trace_json: Optional[str] = None
    trace_md: Optional[str] = None
    error: Optional[str] = None


def _parse_step(obj: dict) -> AutomationStep:
    action = str(obj.get("action", "click")).lower()
    return AutomationStep(
        action=action,
        intent=obj.get("intent"),
        value=obj.get("value"),
        selector=obj.get("selector"),
        selector_type=str(obj.get("selector_type", "auto")),
        dom_node_id=obj.get("dom_node_id"),
        timeout_ms=int(obj.get("timeout_ms", 10000)),
        clear_first=bool(obj.get("clear_first", True)),
        press_enter=bool(obj.get("press_enter", False)),
        wait_after_ms=int(obj.get("wait_after_ms", 250)),
    )


def _parse_criterion(obj: dict) -> GoalCriterion:
    return GoalCriterion(
        kind=str(obj.get("kind", "state_changed")),
        value=str(obj.get("value", "true")),
        required=bool(obj.get("required", True)),
        target_dom_node_id=obj.get("target_dom_node_id"),
        target_role=obj.get("target_role"),
    )


def _parse_task(raw: dict, idx: int) -> Task:
    task_id = str(raw.get("task_id") or raw.get("id") or f"task-{idx}")
    url = raw.get("url") or raw.get("web") or raw.get("site")
    if not url:
        raise ValueError(f"Task {task_id} missing url/web field.")
    url = str(url)
    goal = str(
        raw.get("goal")
        or raw.get("instruction")
        or raw.get("task")
        or raw.get("ques")
        or raw.get("question")
        or ""
    )
    steps_raw = raw.get("steps") or []
    steps = [_parse_step(s) for s in steps_raw]
    criteria_raw = raw.get("goal_criteria")
    criteria = [_parse_criterion(c) for c in criteria_raw] if isinstance(criteria_raw, list) else None
    category = raw.get("category")
    credentials = raw.get("credentials") if isinstance(raw.get("credentials"), dict) else None
    return Task(
        task_id=task_id,
        url=url,
        goal=goal,
        steps=steps,
        goal_criteria=criteria,
        category=category,
        credentials=credentials,
    )


def _resolve_template(value: Optional[str], vars_map: dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    out = value
    for k, v in vars_map.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out


def _apply_credentials(task: Task) -> Task:
    creds = task.credentials or {}
    if not creds:
        return task
    vars_map = {str(k): str(v) for k, v in creds.items()}
    steps: list[AutomationStep] = []
    for s in task.steps:
        steps.append(AutomationStep(
            action=s.action,
            intent=_resolve_template(s.intent, vars_map),
            value=_resolve_template(s.value, vars_map),
            selector=_resolve_template(s.selector, vars_map),
            selector_type=s.selector_type,
            dom_node_id=_resolve_template(s.dom_node_id, vars_map),
            timeout_ms=s.timeout_ms,
            clear_first=s.clear_first,
            press_enter=s.press_enter,
            wait_after_ms=s.wait_after_ms,
        ))
    criteria = None
    if task.goal_criteria is not None:
        criteria = []
        for c in task.goal_criteria:
            criteria.append(GoalCriterion(
                kind=c.kind,
                value=_resolve_template(c.value, vars_map) or "",
                required=c.required,
                target_dom_node_id=_resolve_template(c.target_dom_node_id, vars_map),
                target_role=_resolve_template(c.target_role, vars_map),
            ))
    return Task(
        task_id=task.task_id,
        url=task.url,
        goal=_resolve_template(task.goal, vars_map) or task.goal,
        steps=steps,
        goal_criteria=criteria,
        category=task.category,
        credentials=task.credentials,
    )


def load_tasks(dataset_path: str) -> list[Task]:
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    tasks: list[Task] = []
    if p.suffix.lower() in {".jsonl", ".jl"}:
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            tasks.append(_parse_task(json.loads(line), i))
        return tasks

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("tasks", [])
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list or an object with `tasks` list.")
        for i, item in enumerate(data, start=1):
            tasks.append(_parse_task(item, i))
        return tasks

    raise ValueError("Unsupported dataset format. Use .jsonl or .json")


def run_tasks(
    dataset_path: str,
    output_dir: str,
    headless: bool = True,
    use_llm_selector: bool = True,
    limit: Optional[int] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> list[TaskResult]:
    setup_logging(level=log_level, log_file=log_file)
    tasks = load_tasks(dataset_path)
    if limit is not None:
        tasks = tasks[: max(0, int(limit))]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "results.json"

    results: list[TaskResult] = []
    try:
        with BrowserAutomationAgent(headless=headless, use_llm_selector=use_llm_selector) as agent:
            for task_idx, task in enumerate(tasks):
                if task_idx > 0:
                    try:
                        agent.reset_context()
                    except Exception as _reset_err:
                        logger.warning(
                            "Context reset failed for task %s: %s", task.task_id, _reset_err
                        )
                task = _apply_credentials(task)
                t0 = time.perf_counter()
                logger.info(
                    "Starting task %s/%s id=%s url=%s",
                    task_idx + 1, len(tasks), task.task_id, task.url,
                )
                try:
                    if task.steps:
                        goal = TaskGoal(description=task.goal, criteria=task.goal_criteria or [])
                        run = agent.run_task_with_goal(
                            task.url,
                            task.steps,
                            goal if goal.criteria else None,
                        )
                        plan_status = None
                        steps_total = len(run.steps)
                        steps_ok = sum(1 for s in run.steps if s.ok)
                        goal_prob = run.goal_progress.probability if run.goal_progress else None
                        success = run.success
                        trace_json = str(out_dir / f"{task.task_id}_trace.json")
                        trace_payload = {
                            "task_id": task.task_id,
                            "mode": "steps",
                            "url": task.url,
                            "goal": task.goal,
                            "success": success,
                            "steps": [
                                {
                                    "action": s.step.action,
                                    "intent": s.step.intent,
                                    "value": s.step.value,
                                    "ok": s.ok,
                                    "duration_ms": round(s.duration_ms, 2),
                                    "changed": s.changed,
                                    "summaries": s.change_summaries[:20],
                                }
                                for s in run.steps
                            ],
                        }
                        Path(trace_json).write_text(
                            json.dumps(trace_payload, indent=2), encoding="utf-8"
                        )
                        trace_md = None
                        extracted_answer = None
                    else:
                        execution = agent.run_user_goal(task.url, task.goal)
                        report_json = str(out_dir / f"{task.task_id}_trace.json")
                        report_md = str(out_dir / f"{task.task_id}_trace.md")
                        agent.export_plan_trace_report(execution, report_json, report_md)
                        plan_status = execution.plan.status
                        steps_total = len(execution.executed_steps)
                        steps_ok = sum(1 for s in execution.executed_steps if s.ok)
                        goal_prob = None
                        success = execution.success
                        extracted_answer = execution.extracted_answer
                        trace_json = report_json
                        trace_md = report_md

                    duration_ms = (time.perf_counter() - t0) * 1000
                    logger.info(
                        "Task %s done: success=%s steps=%d/%d duration=%.0fms",
                        task.task_id, success, steps_ok, steps_total, duration_ms,
                    )
                    results.append(TaskResult(
                        task_id=task.task_id,
                        url=task.url,
                        goal=task.goal,
                        success=success,
                        duration_ms=duration_ms,
                        plan_status=plan_status,
                        steps_total=steps_total,
                        steps_ok=steps_ok,
                        goal_progress_probability=goal_prob,
                        extracted_answer=extracted_answer,
                        trace_json=trace_json,
                        trace_md=trace_md,
                    ))
                except Exception as exc:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    logger.error("Task %s raised exception: %s", task.task_id, exc, exc_info=True)
                    results.append(TaskResult(
                        task_id=task.task_id,
                        url=task.url,
                        goal=task.goal,
                        success=False,
                        duration_ms=duration_ms,
                        plan_status="failed",
                        steps_total=0,
                        steps_ok=0,
                        goal_progress_probability=None,
                        error=str(exc),
                    ))
                finally:
                    # Persist incremental summary so abrupt process exits still
                    # leave a usable results file.
                    summary_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    finally:
        summary_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        logger.info("Results written to %s", summary_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run integrated browser agent on tasks."
    )
    parser.add_argument("--dataset", required=True, help="Path to tasks (.jsonl or .json).")
    parser.add_argument("--output", required=True, help="Output directory for traces and summary.")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode.")
    parser.add_argument("--disable-llm-selector", action="store_true", help="Disable model selector path.")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N tasks.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument("--log-file", default=None, help="Optional file path for log output.")
    args = parser.parse_args()

    results = run_tasks(
        dataset_path=args.dataset,
        output_dir=args.output,
        headless=not args.headed,
        use_llm_selector=not args.disable_llm_selector,
        limit=args.limit,
        log_level=args.log_level,
        log_file=args.log_file,
    )
    success = sum(1 for r in results if r.success)
    print(f"Completed {len(results)} tasks | success={success} | failed={len(results) - success}")


if __name__ == "__main__":
    main()
