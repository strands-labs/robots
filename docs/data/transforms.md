---
description: Multiply a recorded LeRobotDataset with generative episode augmentation - video streams transformed, actions byte-identical, provenance-marked, pixel-verdict re-validated.
---

# Dataset transforms (episode augmentation)

A **dataset transform** is the third provider shape in `strands-robots`,
beside policies (inference) and trainers (post-tuning): LeRobotDataset in,
augmented LeRobotDataset out. It is the one data-diversity mechanism that
escapes the simulator's visual distribution - domain randomization, photoreal
backdrop compositing and train-time color jitter all *perturb within* it - and
the one that multiplies data without robot or sim time: record N episodes,
synthesize visually diverse variants, train on many times N.

```
record (strands-robots)  ->  transform (this page)  ->  train (create_trainer)
```

## The contract

1. **Video streams are transformed; everything else passes through
   byte-identical.** A backend sees only the pixel streams; the surface copies
   the action, state and task columns from the source episode unchanged. A
   generated episode is the *same trajectory* rendered differently, never a
   different trajectory.
2. **Provenance is mandatory.** Every generated episode is recorded in the
   output dataset's `meta/provenance.json` with `synthetic=true`, the source
   episode index, and the transform's name and version. Training filters and
   evaluation read it via `load_provenance()` / `synthetic_episode_indices()`,
   so generated pixels are treated honestly - silent mixing of generated and
   recorded data is the failure mode this field exists to prevent.
3. **Re-validation is the acceptance gate.** Supply a deterministic verdict
   function and every generated episode is re-scored against its source
   episode's verdict; a generated episode that flips the verdict is discarded
   and **counted** (`TransformResult.episodes_discarded`) - measured, not
   assumed. An ungated run says so (`revalidated=False`), so it never
   masquerades as a gated one. The gate's entire discriminating power lives
   in the image columns: contract item 1 holds every other column
   byte-identical, so a verdict that reads no `observation.images.*` column
   returns the same answer on the source and on every variant and can never
   flip. The surface measures which columns the verdict consulted, and a run
   whose verdict read no image column is reported as ungated
   (`revalidated=False`, with the cause in `message`) rather than as a clean
   gated pass. Any way the verdict reads a column counts - subscripting it,
   `get()`, iterating `items()` / `values()`, comparing the whole mapping, or
   taking a copy first (`dict(episode)`, `{**episode}`, `episode.copy()`) -
   because each of those hands the verdict the pixel values, and the
   accusation is only honest when the measurement saw everything the verdict
   saw. Only reads that reach no value at all (`keys()`, iterating keys, `in`,
   `len()`) leave a verdict pixel-blind.

## Usage

```python
from strands_robots.transforms import create_transform, TransformSpec

transform = create_transform("mock")  # the no-dependency reference backend
spec = TransformSpec(
    source_root="/data/recorded",       # what stop_recording() produced
    output_root="/data/augmented",
    variants_per_episode=4,             # N episodes -> up to 4N generated
    seed=7,                             # deterministic per-(episode, variant)
)
problems = transform.validate(spec)     # pure preflight, nothing read/written
if not problems:
    result = transform.transform(spec)
    print(result.episodes_written, result.episodes_discarded)
```

Gate the output with a deterministic verdict function. The verdict must read
at least one `observation.images.<cam>` column - a state- or action-only
predicate cannot flip, because those columns are byte-identical on every
variant, and such a run is reported as ungated:

```python
def verdict(episode) -> bool:
    # episode: {"action": (T, N) float32, "observation.state": (T, N) float32,
    #           "observation.images.<cam>": (T, H, W, 3) uint8, "task": [str, ...]}
    return episode["observation.images.cam"].mean() < 50.0

spec = TransformSpec(
    source_root="/data/recorded",
    output_root="/data/augmented",
    revalidate=verdict,                 # flip -> discard + count
)
```

Filter generated episodes at training / evaluation time:

```python
from strands_robots.transforms import synthetic_episode_indices

synthetic = synthetic_episode_indices("/data/augmented")
# everything in `synthetic` was generated; everything outside it was recorded
```

## Backends

| Provider | What it does | Needs |
|---|---|---|
| `mock` | Deterministic per-variant brightness shift - the reference implementation and test double | nothing |
| `cosmos_transfer` | Cosmos-Transfer-style video2video generation behind a vendor-neutral pipeline seam | a generation pipeline (below) |

### `cosmos_transfer` and the pipeline seam

NVIDIA's Cosmos-Transfer family is the namesake and the intended first
pipeline, but its models ship from source
([github.com/nvidia-cosmos](https://github.com/nvidia-cosmos)) under the
NVIDIA Open Model License - not from PyPI - and their availability and
licensing must be verified per deployment. The backend therefore assumes no
single vendor's model: it binds any object satisfying a small protocol,

```python
class VideoToVideoPipeline:
    def generate(self, video, prompt="", seed=None):
        """(T, H, W, 3) uint8 in -> same shape and dtype out."""
```

supplied either constructed or as a dotted import path resolved lazily:

```python
from strands_robots.transforms import create_transform, TransformSpec

transform = create_transform("cosmos_transfer", pipeline="my_pkg.cosmos:PIPELINE")
spec = TransformSpec(
    source_root="/data/recorded",
    output_root="/data/augmented",
    prompt="the same scene in a cluttered kitchen at night",
    variants_per_episode=4,
    seed=7,
)
```

Without a pipeline, `validate()` names the missing seam (and the licensing
caveat) instead of crashing; nothing is read or written.

## Custom backends

Subclass `DatasetTransform`, implement `provider_name`, `validate` (call
`self._spec_problems(spec)` first) and `transform_frames`, then register:

```python
from strands_robots.transforms import register_transform

register_transform("my_v2v", lambda: MyTransform)
```

The base class owns the dataset plumbing, pass-through, provenance and
re-validation gate, so a backend cannot accidentally weaken them.
