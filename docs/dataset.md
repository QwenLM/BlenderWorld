# Dataset layout

The full bundle is available at [Qwen/BlenderWorld](https://huggingface.co/datasets/Qwen/BlenderWorld). Its `test` split contains all 223 tasks; `all`, `text`, `image`, and `video` are viewer configurations, not training splits.

```text
BlenderWorld/
  README.md
  dataset.jsonl
  rubrics.json
  annotation_notes.json
  metadata/{text,image,video}.{jsonl,parquet}
  manifest.json
  SHA256SUMS
  tasks/bw_0001/
    task.json
    query.txt
    rubric.json
    references/
    attachments/       # when declared
    supplemental/      # additional source files, when present
```

`dataset.jsonl` contains one full task record per line. `task.json` is the same record. Each resource has a path relative to the dataset root, a `task_relative_path`, byte size, and SHA256. `query_file` and `rubric_file` are dataset-root-relative. Paths inside `rubric.json` are task-directory-relative.

| Field | Meaning |
| --- | --- |
| `task_id` | Stable neutral identifier |
| `modality` | `text`, `image`, or `video` |
| `query` | Generation request |
| `reference_images`, `reference_videos` | Generation reference media |
| `attachments` | Declared specification inputs |
| `supplemental_attachments` | Additional files stored beside declared inputs; not staged by `bw prepare` |
| `rubric_reference_images` | Separate oriented/resized reference variant, where the frozen rubric used one |
| `rubric_count` | Number of binary criteria |
| `rubric_prompt_matches_query` | Whether the rubric prompt matches the generation request |
| `metadata` | Language, domain, scenario, and original task descriptors |

Every rubric criterion includes `id`, `criterion`, `pass_condition`, `fail_condition`, `importance_weight`, and `evidence_type`. A `scene_summary` criterion requires scene inspection; `render` and `either` criteria use visual judging. Two records retain logical `clarification_markers`; these are not filesystem paths.

`metadata/*.parquet` provides typed viewer rows with query text, resource paths, task/rubric paths, and counts; the equivalent JSONL files are included for direct inspection. Explicit list types allow text tasks with empty media lists to load alongside image and video tasks. To access complete annotations locally:

```python
import json
from pathlib import Path

root = Path("data/BlenderWorld")
task = json.loads((root / "tasks/bw_0001/task.json").read_text())
rubric = json.loads((root / task["rubric_file"]).read_text())
```

The full index is also accessible through Hugging Face Datasets:

```python
from datasets import load_dataset
tasks = load_dataset("Qwen/BlenderWorld", "all", split="test")
```

Download the file bundle for evaluation; viewer rows expose paths, not the files themselves. `bw verify` validates all resource hashes, task/query/rubric consistency, criterion counts, and the release checksum inventory.
