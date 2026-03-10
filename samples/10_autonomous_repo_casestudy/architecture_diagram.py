#!/usr/bin/env python3
"""
Architecture Diagram Generator for Autonomous Agent System

Generates text-based architecture diagrams showing the autonomous feedback loop.
Part of Sample 10: Autonomous Repository Case Study (K12 curriculum).
"""

from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _require_yaml():
    """Raise a clear error if pyyaml is not installed."""
    if yaml is None:
        raise ImportError(
            "pyyaml is required for YAML parsing features: pip install pyyaml"
        )


def generate_feedback_loop_diagram() -> str:
    """Generate ASCII art diagram of the autonomous feedback loop."""
    diagram = """
# Autonomous Agent Architecture: Feedback Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS DEVELOPMENT CYCLE                     │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────┐
    │  GitHub  │
    │  Issue   │──────┐
    │  #N      │      │
    └──────────┘      │
         │            │
         │ Trigger    │
         ▼            │
    ┌──────────────┐  │
    │   Workflow   │  │
    │  (agent.yml) │  │
    │              │  │
    │ • Parse issue│  │
    │ • Build      │  │
    │   context    │  │
    └──────────────┘  │
         │            │
         │ Invoke     │
         ▼            │
    ┌──────────────┐  │
    │  Knowledge   │  │
    │  Base (RAG)  │  │
    │              │  │
    │ • Retrieve   │──┘ Context enrichment
    │   relevant   │
    │   memories   │
    └──────────────┘
         │
         │ Enhanced context
         ▼
    ┌──────────────┐
    │ Strands Agent│
    │  (Claude)    │
    │              │
    │ • Analyze    │
    │ • Plan       │
    │ • Execute    │
    └──────────────┘
         │
         │ Tool calls
         ▼
    ┌──────────────────────────────────┐
    │         Available Tools          │
    ├──────────────────────────────────┤
    │ • shell (code execution)         │
    │ • file_read/write (editing)      │
    │ • use_github (PR/issue ops)      │
    │ • store_in_kb (memory storage)   │
    │ • retrieve (memory retrieval)    │
    │ • projects (project board mgmt)  │
    │ • scheduler (recurring tasks)    │
    └──────────────────────────────────┘
         │
         │ Generates
         ▼
    ┌──────────────┐
    │  Code & PRs  │
    │              │
    │ • New files  │
    │ • Tests      │
    │ • Docs       │
    └──────────────┘
         │
         │ Creates/Updates
         ▼
    ┌──────────────┐
    │  Pull Request│
    │  (optional)  │
    │              │
    │ • Review     │
    │ • CI/CD      │
    └──────────────┘
         │
         │ Merge
         ▼
    ┌──────────────┐
    │  Main Branch │
    │              │
    │ • Production │
    │   code       │
    └──────────────┘
         │
         │ Store outcome
         ▼
    ┌──────────────┐
    │  Knowledge   │
    │  Base (RAG)  │
    │              │
    │ • Store      │
    │   results    │
    │ • Learn      │
    └──────────────┘
         │
         │ Informs future decisions
         └─────────────────┐
                           │
                           ▼
                      ┌──────────┐
                      │   Next   │
                      │  Issue   │
                      │  #N+1    │
                      └──────────┘

```

## Key Components

### 1. Trigger Mechanisms
- **Issue Events:** `opened`, `edited`, `labeled`
- **PR Events:** `opened`, `review_requested`
- **Comment Events:** `created` on issues/PRs
- **Schedule:** Cron-based recurring tasks
- **Manual:** `workflow_dispatch`

### 2. Context Building
The agent receives rich context including:
- Issue/PR body and full comment thread
- Project board state (via STRANDS_CODER_PROJECT_ID)
- Retrieved memories from knowledge base (semantic search)
- System prompt with self-awareness (own source code)

### 3. Autonomous Execution
The agent can:
- Read and write files
- Execute shell commands (git, python, tests)
- Create/update issues and PRs
- Manage project board items
- Schedule future tasks

### 4. Memory & Learning
- **store_in_kb:** Stores interaction outcomes in AWS Bedrock Knowledge Base
- **retrieve:** Semantic search over past interactions
- **Enables:** Cross-session context, learning from past mistakes

### 5. Continuous Improvement
Each cycle generates data that improves future decisions through RAG.
"""
    return diagram


def analyze_workflows() -> str:
    """Analyze workflow files and their triggers."""
    _require_yaml()

    workflows_dir = Path(".github/workflows")
    if not workflows_dir.exists():
        return "No workflows directory found."

    output = "\n# Workflow Files and Triggers\n\n"

    workflow_files = sorted(workflows_dir.glob("*.yml"))

    for workflow_file in workflow_files:
        try:
            with open(workflow_file, "r") as f:
                content = yaml.safe_load(f)

            name = content.get("name", workflow_file.stem)
            triggers = content.get("on", {})

            output += f"## {name}\n"
            output += f"**File:** `.github/workflows/{workflow_file.name}`\n\n"

            if isinstance(triggers, dict):
                output += "**Triggers:**\n"
                for trigger, config in triggers.items():
                    if trigger == "schedule":
                        if isinstance(config, list):
                            for schedule in config:
                                cron = schedule.get("cron", "N/A")
                                output += f"- `{trigger}`: `{cron}`\n"
                        else:
                            output += f"- `{trigger}`\n"
                    elif trigger == "workflow_dispatch":
                        output += f"- `{trigger}` (manual)\n"
                        if isinstance(config, dict) and "inputs" in config:
                            inputs = config["inputs"]
                            output += "  - **Inputs:**\n"
                            for input_name, input_config in inputs.items():
                                desc = input_config.get("description", "")
                                output += f"    - `{input_name}`: {desc}\n"
                    elif isinstance(config, dict):
                        types = config.get("types", [])
                        if types:
                            output += f"- `{trigger}`: {', '.join(types)}\n"
                        else:
                            output += f"- `{trigger}`\n"
                    else:
                        output += f"- `{trigger}`\n"
            elif isinstance(triggers, list):
                output += "**Triggers:**\n"
                for trigger in triggers:
                    output += f"- `{trigger}`\n"
            else:
                output += f"**Triggers:** `{triggers}`\n"

            # Extract job info
            jobs = content.get("jobs", {})
            if jobs:
                output += f"\n**Jobs:** {len(jobs)} job(s)\n"
                for job_name in jobs.keys():
                    output += f"- `{job_name}`\n"

            output += "\n"

        except Exception as e:
            output += f"**Error parsing {workflow_file.name}:** {e}\n\n"

    return output


def analyze_scheduler() -> str:
    """Analyze scheduled jobs from scheduler workflow."""
    _require_yaml()

    scheduler_file = Path(".github/workflows/scheduler.yml")

    if not scheduler_file.exists():
        return "\n# Scheduled Jobs\n\nNo scheduler.yml found.\n"

    try:
        with open(scheduler_file, "r") as f:
            content = yaml.safe_load(f)

        output = "\n# Scheduled Jobs (scheduler.yml)\n\n"

        triggers = content.get("on", {})
        schedules = triggers.get("schedule", [])

        if schedules:
            output += "## Cron Schedules\n\n"
            for idx, schedule in enumerate(schedules, 1):
                cron = schedule.get("cron", "N/A")
                output += f"{idx}. `{cron}`\n"
            output += "\n"

        jobs = content.get("jobs", {})
        if jobs:
            output += "## Jobs\n\n"
            for job_name, job_config in jobs.items():
                output += f"### {job_name}\n"

                steps = job_config.get("steps", [])
                output += f"**Steps:** {len(steps)}\n\n"

                for step in steps:
                    step_name = step.get("name", "(unnamed)")
                    output += f"- {step_name}\n"

                output += "\n"

        return output

    except Exception as e:
        return f"\n# Scheduled Jobs\n\nError parsing scheduler.yml: {e}\n"


def generate_tool_diagram() -> str:
    """Generate diagram showing tool ecosystem."""
    diagram = """
# Agent Tool Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                     STRANDS AGENT TOOLS                      │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  File Operations │   │  Code Execution  │   │  GitHub Actions  │
├──────────────────┤   ├──────────────────┤   ├──────────────────┤
│ • file_read      │   │ • shell          │   │ • use_github     │
│ • file_write     │   │   - git commands │   │   - Issues       │
│ • editor         │   │   - python run   │   │   - PRs          │
│                  │   │   - tests        │   │   - Projects     │
│                  │   │   - build tools  │   │   - GraphQL API  │
└──────────────────┘   └──────────────────┘   └──────────────────┘

┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Memory (RAG)    │   │  Project Mgmt    │   │  Automation      │
├──────────────────┤   ├──────────────────┤   ├──────────────────┤
│ • store_in_kb    │   │ • projects       │   │ • scheduler      │
│   - AWS Bedrock  │   │   - Create items │   │   - Recurring    │
│ • retrieve       │   │   - Update status│   │     tasks        │
│   - Semantic     │   │   - Query board  │   │   - Cron jobs    │
│     search       │   │                  │   │                  │
└──────────────────┘   └──────────────────┘   └──────────────────┘

                    ┌──────────────────┐
                    │  Sub-Agents      │
                    ├──────────────────┤
                    │ • create_subagent│
                    │   - Specialized  │
                    │     agents       │
                    │   - Parallel     │
                    │     execution    │
                    └──────────────────┘
```

## Tool Capabilities

### File Operations
- **Read:** View files, search patterns, get stats
- **Write:** Create/update files with validation
- **Editor:** Advanced editing with diff preview

### Code Execution (shell)
- Git operations (commit, push, PR)
- Python script execution
- Test running (pytest)
- Build commands (ruff, mkdocs)
- System commands

### GitHub Actions (use_github)
- Issue management (create, update, comment)
- PR operations (create, review, merge)
- Project board management (V2 API)
- GraphQL queries (flexible API access)

### Memory & RAG
- **store_in_kb:** Persist interaction outcomes
- **retrieve:** Semantic search over past context
- **Use case:** Cross-session learning, avoiding repeated mistakes

### Project Management
- Query project board state
- Update item status (Todo → In Progress → Done)
- Create draft issues
- Link items to issues/PRs

### Automation
- Schedule recurring tasks
- Define cron expressions
- Automated maintenance (dependency updates, test runs)
"""
    return diagram


def main():
    """Main entry point."""
    print("Generating architecture diagrams...\n")

    output = "# Autonomous Agent Architecture Documentation\n"
    output += "**Generated for Sample 10: Autonomous Repository Case Study**\n\n"

    # Add feedback loop diagram
    output += generate_feedback_loop_diagram()

    # Add workflow analysis
    output += analyze_workflows()

    # Add scheduler analysis
    output += analyze_scheduler()

    # Add tool ecosystem diagram
    output += generate_tool_diagram()

    # Write to file
    output_path = Path(__file__).parent / "ARCHITECTURE.md"
    output_path.write_text(output)

    print("✅ Architecture documentation generated!")
    print(f"📄 Written to: {output_path}\n")
    print("="*70)
    print(output[:2000] + "..." if len(output) > 2000 else output)
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
