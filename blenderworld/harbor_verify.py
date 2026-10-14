"""Harbor verifier entry point, backed by the standalone BlenderWorld evaluator."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .data import load_tasks, sha256, write_json
from .evaluation import PROTOCOL, evaluate, tree_hashes
from .submissions import pack_submission


def run_verifier(dataset, submission, output):
    dataset, submission, output = map(lambda p: Path(p).resolve(), (dataset, submission, output))
    if output == submission or output.is_relative_to(submission):
        raise ValueError("Verifier output must be outside the submission")
    output.mkdir(parents=True, exist_ok=True)
    reward = {"reward": 0.0, "rubric_score": 0.0, "dimension_score": 0.0}
    # A timeout/crash can never leave a previous successful reward behind.
    write_json(output / "reward.json", reward)
    result = {"ok": False, "status": "running", "score": 0.0, "protocol": PROTOCOL}
    write_json(output / "metrics.json", result)
    try:
        rows = load_tasks(dataset)
        if len(rows) != 1:
            raise ValueError("A Harbor verifier package must contain exactly one task")
        task_id = rows[0]["task_id"]
        result["task_id"] = task_id
        if submission.is_symlink() or any(p.is_symlink() for p in submission.rglob("*")):
            raise ValueError("Submission contains symbolic links")
        before = sha256(submission) if submission.is_file() else tree_hashes(submission)
        staging = output / "submissions"
        staging.mkdir(exist_ok=False)
        if submission.is_dir():
            pack_submission(submission, task_id, staging / (task_id + ".zip"))
        elif submission.is_file():
            shutil.copyfile(submission, staging / (task_id + ".zip"))
        options = SimpleNamespace(
            dataset=str(dataset), submissions=str(staging), output=str(output / "run"),
            task_id=[task_id], dry_run=False, resume=False,
            blender_bin=os.environ.get("BLENDER_BIN", "blender"),
            render_agent_model=os.environ.get("RENDER_AGENT_MODEL", "gpt-5.6-terra"),
            agent_judge_model=os.environ.get("AGENT_JUDGE_MODEL", "gpt-5.6-terra"),
            judge_model=os.environ.get("JUDGE_MODEL", "gpt-5.6-terra"),
            judge_api_endpoint=os.environ.get("JUDGE_API_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
        evaluate(options)
        outcome = json.loads((output / "run/tasks" / task_id / "result.json").read_text())
        after = sha256(submission) if submission.is_file() else tree_hashes(submission)
        if after != before:
            raise ValueError("Original submission changed during verification")
        result.update(outcome, status="scored" if outcome["ok"] else "failed")
        if outcome["ok"]:
            reward = {"reward": outcome["score"] / 100,
                      "rubric_score": outcome["rubric_score"] / 100,
                      "dimension_score": outcome["dimension_score"] / 100}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result.update(ok=False, status="failed", score=0.0, error=str(error))
    write_json(output / "metrics.json", result)
    write_json(output / "reward.json", reward)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/tests/dataset")
    parser.add_argument("--submission", default="/workspace/exports",
                        help="Harbor exports directory (packaged before scoring), or one submission ZIP")
    parser.add_argument("--output", default="/logs/verifier")
    args = parser.parse_args()
    result = run_verifier(args.dataset, args.submission, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
