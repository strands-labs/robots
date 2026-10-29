### Docs: the hello-world example prints the rollout summary, not only its status

`examples/01_sim_hello_world.py` ended on `print(f"run_policy status:
{result['status']}")`, so its whole output was the word `success` beside the
instruction `"pick up the red cube"` -- which reads as a pick that happened. It
did not: `MockPolicy` drives a fixed per-joint test motion and does not read the
instruction, and the rollout says so in `result["content"][0]["text"]`, the same
summary an agent mounting the simulation as a tool receives. The example now
prints that block too, so the first thing a reader runs shows the policy and
instruction, the step count and sim time, and the note that the words shaped
nothing -- the honest report the bare status hid. The docstring's expected
output is updated to match.
