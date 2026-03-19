---
name: create-skill-tester
description: Demonstrates and validates the `/create-skill` workflow for creating project-level Cursor skills. Use when testing skill authoring, naming, and structure for `.cursor/skills` assets.
---

# Create-skill Tester

## Purpose

This temporary skill is for validating skill creation in the current repository. It demonstrates the expected structure for a project skill and can be used as a smoke test during onboarding or workflow checks.

## Quick Start

1. Create a directory at `.cursor/skills/<skill-name>/`.
2. Add a `SKILL.md` file with the required frontmatter (`name` and `description`).
3. Keep the skill concise, with clear trigger conditions and predictable instructions.

## Core Pattern

- Put the key behavior and prerequisites in `SKILL.md`.
- Keep optional reference content in one optional side file under the same directory.
- Avoid creating deep dependency chains across many files for simple skills.
- Use plain markdown and ASCII text unless special characters are required.

## Minimal Sections

- `Instructions`: step-by-step flow the agent should follow.
- `Examples`: one short example with input and expected output.
- `Verification`: optional checklist or commands to confirm results.

## Success Criteria

- A valid `.cursor/skills/.../SKILL.md` exists in the repository.
- Frontmatter keys are present and parse cleanly.
- The skill describes the WHEN and WHAT clearly.

