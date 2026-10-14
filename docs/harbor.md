# Run BlenderWorld with Harbor

BlenderWorld exports 223 self-contained Harbor tasks from the same frozen dataset
used by `bw evaluate`. The export includes the generation environment, Blender MCP,
declared inputs, an independent verifier, and content-hashed dataset/task manifests.
Exporting creates local files only. Harbor is optional; the public submission format and standalone evaluator work with any generation harness.

## Export locally

Use Python 3.12+ for Harbor; the standalone BlenderWorld CLI still supports Python
3.10+. Install from the code repository and point at the local HF file bundle:

```bash
python -m pip install -e '.[harbor]'
bw export-harbor --dataset ../BlenderWorld-data --output ../BlenderWorld-harbor \
  --namespace Qwen
```

The namespace is a local package label; it does not claim the Harbor organization.
Choose an organization you can publish to when preparing the final release.
The exporter refuses to overwrite an existing directory. Use `--task-id bw_0001`
(repeatable) for a small selection. `--cpus`, `--memory-mb`, `--gpus`,
`--agent-timeout`, and `--verifier-timeout` set resource budgets. Defaults are
8 cores, 16 GiB RAM, no required GPU, four hours for generation and eight hours
for verification. Both environments require Linux x86-64.

```text
BlenderWorld-harbor/
  dataset.toml
  export_manifest.json
  SHA256SUMS
  verifier.env.example
  bw_0001/
    task.toml
    instruction.md
    environment/     # public image recipe, MCP launcher, bundled generation inputs
    tests/           # independent verifier image, evaluator code, rubric and media
```

Queries and input bytes are preserved. Supplemental files and rubric-specific
image variants stay in the verifier. The package uses Harbor 0.23.0's schema and
official content hashing. Each task has a neutral name such as
`Qwen/BlenderWorld-bw_0001`. All tasks are independent and can be downloaded alone.

## Run

Install Docker and ensure its daemon is running, or configure a compatible Harbor
sandbox. The images build from public Python/Node distributions and install Blender
4.2.10, Node.js 22.17.0, Codex CLI 0.147.0, Claude Code 2.1.220, and MCP for Blender 2.1.3. Optional add-on packs from other execution setups
are not assumed to exist; report this generation environment when comparing runs.

Configure the four verifier endpoint/key variables in `verifier.env.example` and
your generation agent's own credentials. Endpoint-specific model aliases are
configurable. Verifier credentials are passed only in the verifier phase.

```bash
harbor run -p ../BlenderWorld-harbor/bw_0001 -a codex --ak version=0.147.0 -m <provider/model>
harbor run -p ../BlenderWorld-harbor -a codex --ak version=0.147.0 -m <provider/model>
```

Always pass `--ak version=0.147.0` for Codex. For Claude Code, use `-a claude-code --ak version=2.1.220`. Harbor can otherwise install a different CLI version even when the image is pinned.

The agent receives `/workspace/inputs/` and writes its `.blend`, requested exports,
and relative texture dependencies to `/workspace/exports/`. A task MCP server is
available to compatible agents; the Blender CLI is also installed. Harbor restores
the entire exports directory to that same path in a new verifier container.
The verifier packages these deliverables as `/logs/verifier/submissions/<task_id>.zip`, then calls the same ZIP-based BlenderWorld evaluator used by standalone scoring. See [the submission ZIP contract](submissions.md) for rescoring the resulting standard ZIP. Merely creating a
`.blend` file does not earn a passing score.

For local runs, pass the directory, not `dataset.toml`: Harbor 0.23.0 discovers
immediate task subdirectories. The adjacent manifest pins the versions used by
publication. Run `sha256sum -c SHA256SUMS` in the export directory before evaluating
an archived package. The root layout also lets `harbor publish` discover all tasks
alongside the manifest without any remote task dependencies.

To score downloaded outputs separately, select the directory containing `final.blend`
and package it using the public format:

```bash
bw pack-submission --source path/to/trial/artifacts/workspace/exports \
  --task-id bw_0001 --output submissions/bw_0001.zip
```

If the outputs are in an archive, extract that archive first and select the folder
containing `final.blend`. The ZIP produced by this command has `final.blend` at its
root; the evaluator does not consume trial logs or require a Harbor directory tree.
The verifier produces the same public ZIP automatically.

The standalone [evaluation protocol](evaluation.md) applies: eight Cycles views,
1024×1024, 64 samples, up to two corrective rerenders per view, structural/visual
criterion routing, eight quality dimensions, equal-weight fusion, and no extra cap.

`/logs/verifier/reward.json` contains three metrics on a 0–1 scale:

```json
{"reward": 0.95, "rubric_score": 1.0, "dimension_score": 0.9}
```

The main benchmark metric is the mean of `reward`; multiply by 100 to compare
with standalone scores. `metrics.json` retains the 0–100 scores and failure status.
`run/` retains model/protocol hashes, views, individual verdicts, and logs. Failed
or missing submissions receive zero; incomplete runs must not be reported as a
completed 223-task result. Harbor averages each reward dimension independently.

## Verify and publish

The export command verifies the source dataset, parses every task with Harbor,
and produces canonical package hashes. Validate a collected submission with
`bw evaluate --dataset <dataset> --submission <task_id>.zip --output <new-run> --dry-run`.

Also build/run representative text, image, and video tasks in the target sandbox
before publication. Schema validation and offline tests do not replace container
or live judge checks. No oracle reference solution is bundled, so do not select
Harbor's oracle agent.

After reviewing a concrete export and deciding to publish, authenticate to the
owning Harbor organization and publish a tagged version:

```bash
harbor auth login
harbor publish ../BlenderWorld-harbor --tag v1 --public
```

Upload jobs and configure a leaderboard separately. Publishing tasks does not
create leaderboard scores, and a leaderboard does not calculate row metrics from
linked trials. Keep dataset/code/model versions with any reported results.
