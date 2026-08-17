### Fixed

- **policies/lerobot_local**: a best-effort Hub read in the processor bridge no longer aborts the policy load when `huggingface_hub` is installed but unimportable. Both reads imported `HfHubHTTPError` inside the same `try` whose handler names it, so any non-`ImportError` import failure raised `UnboundLocalError` while evaluating that handler - masking the real error and skipping the `OSError`/`ValueError` handler written for it. The error class is now imported in its own `try`, and a new package-wide guard refuses the shape.
