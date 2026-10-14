"""Command line interface for local task preparation and evaluation."""
import argparse
import json
import os
from pathlib import Path

from .data import load_tasks, prepare, verify


def main():
    parser = argparse.ArgumentParser(prog="bw", description="BlenderWorld benchmark tools")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the frozen dataset from Hugging Face")
    download.add_argument("--repo-id", default="Qwen/BlenderWorld")
    download.add_argument("--revision", default="main")
    download.add_argument("--output", default="data/BlenderWorld")
    check = commands.add_parser("verify", help="Verify task content, paths and checksums")
    check.add_argument("dataset")
    listing = commands.add_parser("list", help="List tasks")
    listing.add_argument("dataset")
    listing.add_argument("--modality", choices=["text", "image", "video"])
    staging = commands.add_parser("prepare", help="Stage a task's generator-visible inputs")
    staging.add_argument("--dataset", required=True)
    staging.add_argument("--task-id", required=True)
    staging.add_argument("--output", required=True)
    packing = commands.add_parser("pack-submission", help="Package model outputs as one submission ZIP")
    packing.add_argument("--source", required=True, help="Output directory or ZIP with final.blend at its root")
    packing.add_argument("--task-id", required=True)
    packing.add_argument("--output", required=True, help="New submissions/bw_XXXX.zip")
    harbor = commands.add_parser("export-harbor", help="Export self-contained Harbor tasks (Python 3.12+)")
    harbor.add_argument("--dataset", required=True)
    harbor.add_argument("--output", required=True)
    harbor.add_argument("--namespace", default="Qwen")
    harbor.add_argument("--task-id", action="append")
    harbor.add_argument("--cpus", type=int, default=8)
    harbor.add_argument("--memory-mb", type=int, default=16384)
    harbor.add_argument("--gpus", type=int, default=0)
    harbor.add_argument("--agent-timeout", type=int, default=14400)
    harbor.add_argument("--verifier-timeout", type=int, default=28800)
    scoring = commands.add_parser("evaluate", help="Render and judge per-task BlenderWorld submission ZIPs")
    scoring.add_argument("--dataset", required=True)
    scoring.add_argument("--submissions", "--submission", dest="submissions", required=True,
                         help="Directory of bw_XXXX.zip files, or a single task ZIP")
    scoring.add_argument("--output", required=True)
    scoring.add_argument("--task-id", action="append")
    scoring.add_argument("--dry-run", action="store_true")
    scoring.add_argument("--resume", action="store_true")
    scoring.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", "blender"))
    scoring.add_argument("--render-agent-model", default=os.environ.get("RENDER_AGENT_MODEL", "gpt-5.6-terra"))
    scoring.add_argument("--agent-judge-model", default=os.environ.get("AGENT_JUDGE_MODEL", "gpt-5.6-terra"))
    scoring.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gpt-5.6-terra"))
    scoring.add_argument("--judge-api-endpoint", default=os.environ.get("JUDGE_API_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
    args = parser.parse_args()
    try:
        if args.command == "download":
            from huggingface_hub import snapshot_download
            snapshot_download(args.repo_id, repo_type="dataset", revision=args.revision, local_dir=args.output)
            result = verify(args.output)
        elif args.command == "verify":
            result = verify(args.dataset)
        elif args.command == "prepare":
            result = prepare(args.dataset, args.task_id, args.output)
        elif args.command == "pack-submission":
            from .submissions import pack_submission
            result = pack_submission(args.source, args.task_id, args.output)
        elif args.command == "list":
            result = [{"task_id": r["task_id"], "modality": r["modality"], "rubric_count": r["rubric_count"]}
                      for r in load_tasks(args.dataset) if not args.modality or r["modality"] == args.modality]
        elif args.command == "export-harbor":
            from .harbor_export import export_harbor
            result = export_harbor(args.dataset, args.output, args.namespace, args.task_id,
                                   args.cpus, args.memory_mb, args.gpus,
                                   args.agent_timeout, args.verifier_timeout)
        else:
            from .evaluation import evaluate
            result = evaluate(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "evaluate" and not args.dry_run and result["overall"]["succeeded"] != result["overall"]["task_count"]:
            return 2
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"bw: {error}\n")
