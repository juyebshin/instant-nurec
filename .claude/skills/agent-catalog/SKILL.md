---
name: agent-catalog
description: Agent catalogue and model routing for this repo — which agent to spawn for which Instant NuRec task (training, data, HD map, occupancy, simulation, research), and which model tier each should run on. Consult before delegating work with the Agent tool.
user-invocable: false
---

# Agent Catalog & Model Routing

Adapted from `../mapnn/.claude/skills/agent-catalog`. Pick the agent by task, then confirm the tier.

## Model routing policy

| Tier | Use for |
|---|---|
| `haiku` | one-fact lookups, mechanical doc edits |
| `sonnet` | executing a known procedure: implementation, debugging, tracing, data export |
| `opus` | judgement where being wrong is expensive: review, experiment design, interpretation |
| `fable` | knowledge this repo does not contain: literature, published protocols, method comparisons |

Rule of thumb: **if the agent mainly executes a known procedure, sonnet. If it mainly forms a
judgement, opus. If it needs knowledge from outside the repo, fable.** Override per call with
`Agent({ subagent_type: "...", model: "..." })`. The frontmatter `model:` is only the default.

## Project agents (`.claude/agents/`)

| Agent | Model | Writes? | Use for |
|---|---|---|---|
| `inurec-explorer` | sonnet | no | where a value or tensor lives, what a change would touch, including the PMT / mapnn / DVGT / gsdiff call sites |
| `inurec-debugger` | sonnet | yes | crashes, NaN, OOM, skipped losses, metric regressions; 5 phases |
| `data-pipeline-engineer` | sonnet | yes | PMT index exporters (DDAD, KITTI, OpenScene), `PMTIndexDataset`, label alignment |
| `task-head-integrator` | sonnet | yes | a new readout or head for a downstream task (occupancy, map classes), default-off |
| `inurec-reviewer` | opus | no | pre-commit diff review |
| `experiment-scientist` | opus | no | turn a question into a run list; interpret results against the pinned baseline |
| `scene-researcher` | fable | no | literature-grounded design: feed-forward 3DGS, Gaussian occupancy, road reconstruction, simulation; **the only web-enabled agent** |

Escalate `inurec-reviewer` to `fable` when a review turns on modelling correctness rather than code
correctness, for example "is this occupancy readout equivalent to GaussianFormer's aggregation?".

## Routing by task shape

| Task | Agent | Skill it should load |
|---|---|---|
| "add DDAD" / "the depth labels look wrong" | `data-pipeline-engineer` | `inurec-data` |
| "can the training path take varying cameras" | `inurec-explorer` first, then `inurec-debugger` | `multi-rig-training` |
| "the run crashed / NaN / OOM / PSNR dropped" | `inurec-debugger` | `inurec-train-eval`, `inurec-env` |
| "why did road IoU drop from step 1000 to 9000" | `experiment-scientist` | `hdmap-from-gaussians` |
| "get occupancy out of the Gaussians" | `scene-researcher` (design), then `task-head-integrator` | `occupancy-from-gaussians` |
| "export scenes for NuRec / evaluate renders" | `general-purpose` or self | `simulation-export` |
| "how do published methods do X" | `scene-researcher` | — |
| "review before I commit" | `inurec-reviewer` | — |

## Built-in agents

| Agent | Suggested model | Use for |
|---|---|---|
| `Explore` | sonnet | broad read-only fan-out search when the project explorer's map does not cover it |
| `Plan` | opus | implementation strategy for a multi-file change |
| `general-purpose` | sonnet | multi-step work that matches no project agent |

No marketplace or oh-my-claudecode plugins are enabled for this repo, so `oh-my-claudecode:*` and
`cs-*` names do not resolve here, unlike in `../mapnn`.

## Delegation rules

- One agent per independent lane. Launch independent agents in one message.
- Read-only questions go to `inurec-explorer` or `Explore`, never to an agent with write tools.
- Do not spawn an agent for a lookup you can grep yourself.
- Agents that launch GPU jobs must follow the shared-machine rules in `CLAUDE.md`: check
  `nvidia-smi`, run detached with a log, and never pattern-kill.
- The agent's report is not shown to the user. Relay what matters, including which model ran.
