# Evaluation protocol and runtime

## Environment

Use Linux, Python 3.10+, Blender **4.2.10**, Node.js **22.17.0**, ffmpeg/ffprobe, and Claude Code **2.1.220**. Rollout additionally pins Codex CLI **0.147.0** when using Codex. Disable Claude auto-updates with `DISABLE_AUTOUPDATER=1`. The evaluator reads local dataset and submission files and calls the configured model endpoints. A GPU with Cycles support is recommended; CPU Cycles is supported with the same resolution and sample count.

The rendering and scene-inspection agents use the configured Claude Code CLI and its normal credentials. Set `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, and model aliases appropriate for that endpoint. The vision judge uses `JUDGE_API_ENDPOINT`, `JUDGE_API_KEY`, and `JUDGE_MODEL`. Export these variables in your shell; the CLI does not automatically source shell files. Provide endpoint URLs and credentials for your own model services.

Model calls incur costs at the configured providers. Run `--dry-run` to inspect the selected task set and submission inventory first.

## Submission and evidence

Supply one **`submissions/<task_id>.zip`** per task. It contains `final.blend` at its root, alongside requested images/videos and relative texture dependencies. Use `bw pack-submission` to package the model's output directory, or create the ZIP with an ordinary archive tool; see [the ZIP format and commands](submissions.md).

The evaluator extracts only the deliverables into a fresh per-task `.eval_workspace`. It verifies manifest checksums when provided, preserves relative paths, and checks that the ZIP and extracted files remain unchanged. The renderer and scene inspector open `final.blend` as the primary project. Paths escaping the archive, links, duplicate members, missing primary projects, and invalid manifests are rejected. There is a limit of 100,000 archive members and 20 GiB of uncompressed data.

Before any model call, the evaluator checks the actual Blender, Node.js, and Claude Code versions and records them in `runtime.json`. A version mismatch aborts the run as a configuration error. Packaging an older rollout does not convert its Blender version: generate new submissions with 4.2.10 for a run with matching generation and evaluation versions.

The rendering agent selects eight views from the actual scene. Camera position, framing, clipping, and evaluator render settings may change. Geometry, materials, textures, visibility, lighting, animation, and color management must remain unchanged. The renderer opens every PNG and checks it. It may correct each view at most twice after its initial attempt. A Blender startup hook records actual engine, samples, device, and render completion, and rejects a fourth attempt at the same output path. The output validator requires all eight 1024 × 1024 PNGs and matching runtime records.

Missing texture paths can be restored only from the same submitted workspace, using unambiguous path matches. Saved projects are not rewritten. Original OptiX denoising may use CPU OpenImageDenoise for compatibility. Authored visual defects are retained.

Image references are sent to the judges directly. Video references are sampled into four ordered frames; authored videos provide up to sixteen ordered frames to the visual judge. Declared files and the native project are available to the scene inspector. Author images are selected separately from evaluator views so that final-image requirements can be judged against the authored deliverables.

## Judging and scoring

The scene inspector handles `scene_summary` criteria and saves its raw judgement, inspection evidence, and CLI logs. The vision judge handles `render` and `either` criteria, blind descriptions, and eight holistic dimensions. Each criterion is routed once. Missing structural verdicts fail evaluation instead of being substituted with visual guesses.

Binary rubric scores are weighted by `importance_weight`. The dimension score is ten times the mean of all eight 1–10 ratings. The final score is their unweighted mean, with no catastrophic-failure or scene-fidelity cap. Incomplete dimension output and missing evaluator renders are engineering failures. Failed tasks receive zero in the aggregate; successful-only averages do not replace the benchmark mean.

The default model for all roles is `gpt-5.6-terra`, matching the paper. Endpoint-specific aliases can be supplied with `--render-agent-model`, `--agent-judge-model`, and `--judge-model`. Record these aliases when reporting results.

## Runs and recovery

`plan.json` pins code hashes, task and rubric hashes, original submission hashes, model names, and the protocol. Resuming requires an exact match. A process lock prevents simultaneous owners of the same run.

```bash
bw evaluate --dataset data/BlenderWorld --submissions submissions \
  --output runs/full --resume
```

Completed outcomes are retained, including failures. A run interrupted before a task outcome is recorded saves its partial evidence under `incomplete/` and starts a fresh attempt for that task. To retry terminal failures or change code, inputs, or models, choose a new output directory. No background job is restarted by an observation timeout.

`summary.json` contains overall, text, and multimodal aggregates. `results.csv` contains task scores and errors. Per-task evidence is under `tasks/<task_id>/`; `result.json` records the final status. A partial aggregate has `complete=false`, and pending tasks remain in the selected denominator.
