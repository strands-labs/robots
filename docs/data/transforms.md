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

   Schema parity is part of that promise, down to how each camera is stored.
   The output dataset declares one `dtype` for every camera, so a source whose
   `observation.images.*` streams disagree - one `video`, one `image` - cannot
   be reproduced: writing it anyway would re-encode a camera, flattening a
   video stream into still frames or promoting an image column to video, and
   which way it went would depend only on the order the features were declared
   in. Such a source is refused, naming each camera and the dtype it declared;
   re-record or convert it so every camera stream shares one dtype.

   The same promise decides how the state and action columns are read. Their
   width in the output is the number of names the source declares for them, so
   those names have to describe the vector they annotate. LeRobot writes them
   either as a flat list (`["shoulder_pan", ...]`) or as a mapping that groups
   the components (`{"motors": [...]}` from `teleop_keyboard`, `{"left": [...],
   "right": [...]}` for a bimanual arm, `{"delta_x": 0, ...}` from
   `teleop_gamepad`); all of those are read, so a dataset recorded through any
   LeRobot teleoperator passes through whole. A declaration whose count still
   disagrees with the column's width is refused rather than applied, naming both
   counts: names short of the width would drop the trailing components, and
   names past it would declare a column no frame supplies - written as `0.0`,
   which is itself a travel-to-zero command for an absolute-position actuator.
   Both are silent, and neither output would be the source rendered differently.
2. **Provenance is mandatory.** Every generated episode is recorded in the
   output dataset's `meta/provenance.json` with `synthetic=true`, the source
   episode index, and the transform's name and version. Training filters and
   evaluation read it via `load_provenance()` / `synthetic_episode_indices()`,
   so generated pixels are treated honestly - silent mixing of generated and
   recorded data is the failure mode this field exists to prevent. A record that
   cannot answer what it is read for is refused rather than stored or trusted:
   `episode_index` must be a non-negative whole number and `synthetic` must be a
   boolean, checked by one rule that both `write_provenance()` and
   `load_provenance()` consult. `synthetic` is not coerced, because every
   non-empty string and every non-zero number is truthy - guessing which of them
   meant "generated" is how a generated episode ends up counted as recorded. The
   descriptive keys (`transform_version`, `prompt`, `seed`) are carried through
   untouched; nothing reads them as a verdict.
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

An empty set is that statement, not a shrug: a dataset with no
`meta/provenance.json` declares no synthetic episodes (the ordinary state of a
recorded dataset), while a file that is present but unreadable raises. Absence
and corruption are different verdicts, so "outside the set" always means
recorded.

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

The same holds for a pipeline that is named but cannot be loaded. Resolving the
seam runs your code at three points - the module import and attribute lookup,
the zero-arg construction of a class or factory target, and the read of the
object's `generate` surface - and constructing a real generation pipeline loads
weights and touches a device. So a missing optional dependency imported inside
a factory body, an absent driver, absent weights or a malformed config are
reported by `validate()` as problems, and by `transform()` as
`status="error"`, each naming the class and message the pipeline raised. An
operator interrupt (`KeyboardInterrupt`, `SystemExit`) is not a spec problem
and still propagates.

## Custom backends

Subclass `DatasetTransform`, implement `provider_name`, `validate` (call
`self._spec_problems(spec)` first) and `transform_frames`, then register:

```python
from strands_robots.transforms import register_transform

register_transform("my_v2v", lambda: MyTransform)
```

The base class owns the dataset plumbing, pass-through, provenance and
re-validation gate, so a backend cannot accidentally weaken them.

`transform_frames` is called with the determinism key's two per-call inputs,
`source_episode` and `variant`, and owns one obligation of its own: refuse a
value either is not. Both are non-negative whole numbers - together with
`spec.seed` they are spread through one `SeedSequence` by `derive_variant_seed`,
so an unusable value on any of the three names a stream another variant already
owns rather than failing:

```python
from strands_robots.utils import non_negative_whole_number_error

for name, value in (("source_episode", source_episode), ("variant", variant)):
    if text := non_negative_whole_number_error(value, name, "my_v2v.transform_frames"):
        raise ValueError(text)
```

`derive_variant_seed` applies the same rule to all three, so a backend that
always derives a key inherits it - but refuse in `transform_frames` too, because
a backend need not derive one at all (`mock`'s explicit `pixel_shift` mode reads
no key), and because the refusal should name the counter rather than whatever
the pipeline seam happens to complain about first.
