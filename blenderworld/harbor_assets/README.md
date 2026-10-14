# BlenderWorld for Harbor

This directory is a self-contained Harbor export of the frozen BlenderWorld dataset.
`dataset.toml` pins the content digest of every task. Each task bundles its inputs
and the same evaluator used by the standalone `bw evaluate` command.
The included MIT license covers the BlenderWorld adapter and evaluator code.

Use Linux x86-64, Python 3.12+, Harbor 0.23.0, and a working Docker daemon (or a
compatible Harbor sandbox). Images install Blender 4.2.10, Node.js 22.17.0, Codex CLI 0.147.0,
Claude Code 2.1.220, and MCP for Blender 2.1.3 from public distribution services. All reference media and
specification inputs are bundled. The default export requests 8 CPU cores and
16 GiB RAM per environment; Cycles uses CPU when no supported GPU is available.

```bash
python -m pip install harbor==0.23.0
# Configure the verifier variables listed in verifier.env.example.
harbor run -p ./bw_0001 -a codex --ak version=0.147.0 -m <provider/model>
# Run all local task directories (dataset.toml is the publication manifest):
harbor run -p . -a codex --ak version=0.147.0 -m <provider/model>
```

For Claude Code use `-a claude-code --ak version=2.1.220`. Keep the CLI version argument: Harbor may otherwise install a newer version.

The generation agent needs its own provider credentials. Verifier credentials
are separate: `BW_AGENT_JUDGE_ENDPOINT` and `BW_AGENT_JUDGE_API_KEY` configure the
Anthropic-compatible endpoint used by the render/scene-inspection agents;
`BW_VISION_JUDGE_ENDPOINT` and `BW_VISION_JUDGE_API_KEY` configure the vision
Chat Completions endpoint. Set model aliases that exist at those endpoints.

The generation workspace contains `/workspace/inputs/`. Deliver the editable
project as `/workspace/exports/final.blend` with its requested exports and relative
texture dependencies. Harbor transfers `/workspace/exports/` to a fresh verifier
environment. The verifier first packages the deliverables into
`/logs/verifier/submissions/<task_id>.zip`, then scores that ZIP with the standalone
evaluator. Each ZIP contains `final.blend` at its root, the delivered assets, and an
optional `submission.json` task/checksum manifest. This is the public BlenderWorld
submission format, usable with any model harness. Logs and agent configuration
outside the exports directory are excluded. The verifier image contains `/tests/test.sh`, the task's rubric and
reference files, and the evaluation code. They are not installed in the agent image.

The verifier produces eight Cycles views at 1024×1024 and 64 samples, allows two
corrective rerenders per view, then combines weighted binary rubric and eight
holistic quality ratings with equal weights. There is no extra score cap.
`reward.json` contains `reward`, `rubric_score`, and `dimension_score` on a 0–1
scale; `metrics.json` and `run/` retain 0–100 scores, errors, per-criterion reasons,
rendering evidence, and model/protocol records. Missing or failed submissions
receive zero. Use the `reward` mean as the benchmark score; the two component
metrics are reported separately. Complete results must cover all selected tasks.

The verifier timeout defaults to eight hours; the generation timeout is four
hours. Set resource budgets when exporting. Model calls require paid or otherwise
provisioned endpoints. Creating/exporting a task package does not call any models.

`solution/` is intentionally absent: no oracle/reference submission is bundled.
Choose a real generation agent to run these tasks. Package schema/hash validation
does not establish that a sandbox image builds or that a live model evaluation
has passed; consult the accompanying local validation report for executed checks.

To verify the export files locally:

```bash
sha256sum -c SHA256SUMS
```

The namespace in `dataset.toml` is selected at export time. Publishing requires
access to that Harbor organization and a separate publication decision. No upload
is performed by the exporter. See the GitHub repository's `docs/harbor.md` for
export, validation, and publishing instructions.
