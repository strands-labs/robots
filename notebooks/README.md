# strands-robots notebooks

Twelve notebooks, saved without outputs so the repository stays small: run one to see its results. Each one is also
a plain script in [`scripts/`](scripts/) that runs top to bottom in a terminal or pastes into a REPL.

![the notebook path, 00 to 10](media/hero.svg)

| # | notebook | you leave with |
|---|---|---|
| 00 | [quickstart](00_quickstart.ipynb) | an SO-101 in a world: observe, act, a camera, a GIF; the agent as code |
| 01 | [robot and sim](01_robot_and_sim.ipynb) | `Robot("so101")` and `Robot("g1")` in one world: observe, act, set_pose, cameras, `Robot.run`, `Simulation.run` |
| 02 | [recording](02_recording.ipynb) | `run(record=...)` to LeRobot v3, reloaded with `LeRobotDataset` |
| 03 | [agent builds a world](03_agent_builds_a_world.ipynb) | `Agent(tools=[sim.tool, robot.tool])` adds objects and cameras from English; `--repl` for `while True: agent(input("> "))` |
| 04 | [agent drives the robot](04_agent_drives_the_robot.ipynb) | the agent moves the arm, runs a policy by name, presses e-stop; the operator gate on a hardware-like arm |
| 05 | [policies](05_policies.ipynb) | `Policy(...)`: a function, ONNX presets (the G1 stands, the duck walks), a Hub ACT in-process, units |
| 06 | [remote inference](06_remote_inference.ipynb) | the same ACT from a real `lerobot-policy-server` over Zenoh, latency from the policy's ledger, a server killed mid-run |
| 07 | [dashboard](07_dashboard.ipynb) | `Dashboard(sim).serve()`: the world in a browser tab, phone and desktop |
| 08 | [device and safety](08_device_and_safety.ipynb) | the manifest, the safety chain, refusals, estop mid-run, resume, the watchdog |
| 09 | [mesh](09_mesh.ipynb) | discover, invoke, state streams, broadcast estop, signed resume, a robot in a second process, `use_mesh` |
| 10 | [GPU backends](10_gpu_backends.ipynb) | Isaac Sim and MuJoCo-Warp, measured on an L40S (shown, not re-run) |
| 11 | [hub to hardware](11_hub_to_hardware.ipynb) | the companion to the Hugging Face article: an agent over the arm, a LeRobot v3 recording (pushed with `HF_USER`), a Hub ACT by name, the hardware constructor, the arm on a mesh |

## Run them

From the repository root, Python 3.12:

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[all]" ipykernel nbconvert av playwright
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@main"  # 02, 05, 06, 11
playwright install chromium                                                                    # 07
python notebooks/scripts/00_quickstart.py        # or: jupyter lab notebooks/
```

The LeRobot commit is the one `strands_robots/_extras.py` names (LeRobot's PyPI release predates `remote_inference`).
03 and 04 call a model; they use Amazon Bedrock when `AWS_BEARER_TOKEN_BEDROCK` (or AWS credentials) is set.

## Editing

The script is the source; the notebook is generated from it, then executed:

```bash
python notebooks/_nb.py build notebooks/scripts/NN_x.py
jupyter nbconvert --to notebook --execute --inplace notebooks/NN_x.ipynb   # to look at the results
jupyter nbconvert --clear-output --inplace notebooks/NN_x.ipynb            # before committing
```

Commit notebooks without outputs (`tests/test_notebooks_are_light.py` checks). The GIFs and screenshots the
notebooks show are not in git either. They are assets of the
[`media` release](https://github.com/strands-labs/robots/releases/tag/media), where `media/05/wave.gif` is
`05-wave.gif`. Running a notebook writes them under `notebooks/media/` (gitignored). To publish a new one:

```bash
cp notebooks/media/05/wave.gif /tmp/05-wave.gif && gh release upload media /tmp/05-wave.gif --clobber
```

`_nb.py` holds the builder and a few display helpers (GIFs, inline MP4, tables, quiet imports); `scripts/check_nb.py NN_x.ipynb`
lists any error or stderr output. Neither is part of the package. [`examples/agent_builds_a_scene.py`](examples/agent_builds_a_scene.py)
is the one-file agent example the test suite runs.
