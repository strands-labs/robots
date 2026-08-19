### Docs: notebook 05 deploys the checkpoint it trains instead of dead-ending at load

`examples/notebooks/05_streaming_data_loop.ipynb` loaded the freshly-trained
checkpoint into a `policy` variable and never used it, so the notebook claimed
the data loop closed one step before it did. A new deploy cell runs the loaded
checkpoint through `run_policy(policy_object=...)` - the same pattern as
`docs/training/overview.md` and `examples/07_post_tune_any_policy.py` - and a
markdown note shows the agent-driven form, where the prompt must name
`result.checkpoint_dir` so the LLM fills `run_policy(policy_provider=...)`; a
bare "pick up the cube" prompt would not select the trained checkpoint
(`policy_object` is deliberately not LLM-suppliable, #708). Closes #2470.
