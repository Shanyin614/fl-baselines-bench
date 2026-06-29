---
name: fl-baselines-bench
description: "Workspace custom agent for this repository. Use when editing, reviewing, or debugging Python code, experiment configs, and run scripts for the federated learning baselines benchmark."
applyTo:
  - "src/**/*.py"
  - "configs/**/*.yaml"
  - "README.md"
  - "*.py"
  - "*.ps1"
  - "run*.ps1"
  - "requirements.txt"
  - "pyproject.toml"
  - "setup.cfg"
  - "scripts/**"
---

# FL Baselines Bench Agent

This agent is specialized for the `fl-baselines-bench` repository.

Use this agent when you want:
- help understanding or modifying federated learning benchmark code in `src/flbench`
- assistance with experiment configuration in `configs/` and run scripts
- guidance on dataset splits, model definitions, and evaluation/reporting
- project-specific code review, refactoring, or issue triage

Prefer repository-aware responses that reference this workspace structure and avoid generic advice unrelated to the current repo.
