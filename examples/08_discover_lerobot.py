"""Discover the LeRobot API from a Strands agent with the use_lerobot tool.

Goal: show how `use_lerobot` lets an agent (or you) explore everything LeRobot
exposes - registered robots, policies, teleoperators, cameras - and inspect any
class without hardcoding a thing. It reads from LeRobot's own draccus registries,
so the answers track whatever version of LeRobot you have installed.

Runs anywhere: no hardware, no GPU, no Hugging Face credentials. The calls below
invoke the tool's function directly so the example is self-contained; in an agent,
the same tool is selected by natural language ("which policies can I use?").

    uv pip install "strands-robots[sim-mujoco,lerobot]"
    python examples/08_discover_lerobot.py
"""

from strands_robots.tools import use_lerobot

# use_lerobot is a Strands @tool. ``__wrapped__`` is the original undecorated
# function (preserved by functools.wraps), which we call directly so the example
# is self-contained. In an agent, the same tool is selected by natural language.
call = use_lerobot.__wrapped__


def text(result: dict) -> str:
    """Pull the text content out of a tool result."""
    if isinstance(result, dict):
        return "".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
    return str(result)


# 1. Discover everything at once: packages, modules, and every registered config.
print("=" * 70)
print("1. Discovery: what does this LeRobot install expose?")
print("=" * 70)
print(text(call(module="__discovery__", method="list_modules"))[:900])

# 2. List a single registry. The choices come from LeRobot, not from Strands,
#    so they reflect the installed version (here: the policy types you can train
#    or run).
print("\n" + "=" * 70)
print("2. Registry: which policy types are available?")
print("=" * 70)
print(text(call(module="__registry__", method="policies")))

# 3. The same call works for robots, teleoperators, and cameras.
for kind in ("robots", "teleoperators", "cameras"):
    print(f"\n--- registry: {kind} ---")
    print(text(call(module="__registry__", method=kind))[:300])

# 4. Inspect a class without instantiating it. __describe__ returns its methods
#    and signature, useful for finding the right call before you make it.
print("\n" + "=" * 70)
print("3. Describe a class: LeRobotDataset")
print("=" * 70)
print(text(call(module="datasets.lerobot_dataset.LeRobotDataset", method="__describe__"))[:600])

# 5. Resolve a concrete object from a registry name. Here, the policy class
#    behind "act" - the same lookup create_policy() does under the hood.
print("\n" + "=" * 70)
print("4. Resolve the policy class registered as 'act'")
print("=" * 70)
print(text(call(module="policies.factory", method="get_policy_class", parameters={"name": "act"}))[:400])

print("\nDone. In an agent, ask in natural language: "
      "\"List the LeRobot policies I can use, then describe the ACT policy.\"")
