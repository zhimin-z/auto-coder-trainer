# ARIS Skills Reference

Complete reference for all 34 ARIS skills organized by workflow and category. Each skill is a standalone `SKILL.md` file that can be invoked as a Claude Code slash command.

## Workflows

ARIS skills are organized around four core research workflows:

| Workflow | Entry Skill | Description |
|----------|-------------|-------------|
| **1 — Idea Discovery** | `/idea-discovery` | Literature survey → idea generation → novelty check → GPU pilots |
| **1.5 — Experiment Bridge** | `/experiment-bridge` | Implementation + deployment + result collection |
| **2 — Auto Review** | `/auto-review-loop` | Multi-round autonomous review (score 5/10 → 7.5/10+) |
| **3 — Paper Writing** | `/paper-writing` | Narrative → outline → figures → LaTeX → PDF |
| **Full Pipeline** | `/research-pipeline` | Chains Workflows 1 → 1.5 → 2 → 3 end-to-end |

---

## Skill Catalog

### Research & Discovery

| Skill | Invoke | Description |
|-------|--------|-------------|
| **research-lit** | `/research-lit [topic]` | Multi-source literature search (arXiv, Scholar, Zotero, local PDFs). Produces structured lit review with gap analysis. |
| **idea-creator** | `/idea-creator [direction]` | Generate candidate research ideas from identified gaps. Scores novelty, feasibility, and impact. |
| **novelty-check** | `/novelty-check [idea]` | Cross-model novelty verification against existing literature. Uses Codex MCP for independent assessment. |
| **research-review** | `/research-review [paper]` | Structured peer review with scoring (1-10) across originality, soundness, significance, clarity, reproducibility. |
| **research-refine** | `/research-refine [idea]` | Turn vague ideas into problem-anchored proposals with claim-driven reasoning. |
| **comm-lit-review** | `/comm-lit-review [topic]` | Community-focused literature review emphasizing related work positioning. |
| **proof-writer** | `/proof-writer [theorem]` | Rigorous theorem proof drafting with formal verification hooks. |

### Experiment Execution

| Skill | Invoke | Description |
|-------|--------|-------------|
| **experiment-plan** | `/experiment-plan [idea]` | Claim-driven experiment roadmap generation. Identifies baselines, metrics, ablations. |
| **experiment-bridge** | `/experiment-bridge` | Workflow 1.5: implement code → deploy to GPU → collect results. Supports rsync and git sync. Includes GPT-5.4 code review. |
| **run-experiment** | `/run-experiment [config]` | Execute a single experiment on remote GPU. Supports W&B logging, checkpoint management. |
| **monitor-experiment** | `/monitor-experiment [job]` | Track running experiments: GPU utilization, loss curves, ETA. |
| **analyze-results** | `/analyze-results [dir]` | Statistical analysis of experiment results. Significance tests, ablation tables, comparison plots. |
| **dse-loop** | `/dse-loop [space]` | Design-space exploration loop. Sweeps hyperparameters and architecture choices automatically. |

### Paper Production

| Skill | Invoke | Description |
|-------|--------|-------------|
| **paper-plan** | `/paper-plan [report]` | Generate structured paper outline from narrative report. Sections, claims, figure slots. |
| **paper-figure** | `/paper-figure [spec]` | Generate publication-quality matplotlib/tikz figures from data or specifications. |
| **paper-illustration** | `/paper-illustration [desc]` | AI-generated architecture diagrams. Claude plans → Gemini renders → iterative refinement (requires `GEMINI_API_KEY`). |
| **paper-write** | `/paper-write [outline]` | Draft LaTeX paper body from outline. Anti-hallucination citations via DBLP/CrossRef. |
| **paper-compile** | `/paper-compile [tex]` | Compile LaTeX → PDF. Handles bibliography, figures, formatting. |
| **paper-poster** | `/paper-poster [paper]` | Conference poster generation (tcbposter → A0/A1 PDF + PPTX + SVG). Venue-specific colors. |
| **paper-slides** | `/paper-slides [paper]` | Conference presentation slides (beamer → PDF + PPTX). Speaker notes, talk script, Q&A prep. 4 talk types. |
| **mermaid-diagram** | `/mermaid-diagram [desc]` | Generate Mermaid.js diagrams for method flows and architectures. |
| **pixel-art** | `/pixel-art [desc]` | Generate pixel art assets for figures and presentations. |

### Review & Improvement

| Skill | Invoke | Description |
|-------|--------|-------------|
| **auto-review-loop** | `/auto-review-loop [scope]` | Autonomous multi-round review: review → fix → re-review until score threshold or max rounds. Uses Codex MCP. |
| **auto-review-loop-llm** | `/auto-review-loop-llm [scope]` | Review loop using generic LLM MCP server (non-Codex). For alternative model setups. |
| **auto-review-loop-minimax** | `/auto-review-loop-minimax [scope]` | Review loop using MiniMax MCP server. |
| **auto-paper-improvement-loop** | `/auto-paper-improvement-loop [tex]` | Iterative paper quality improvement. Targets specific weaknesses identified by reviewers. |

### Orchestration & Pipeline

| Skill | Invoke | Description |
|-------|--------|-------------|
| **idea-discovery** | `/idea-discovery [direction]` | **Workflow 1**: Chains `research-lit → idea-creator → novelty-check → research-review`. Full idea exploration. |
| **idea-discovery-robot** | `/idea-discovery-robot [direction]` | Fully automated idea discovery (no human checkpoints). |
| **research-pipeline** | `/research-pipeline [direction]` | **Full Pipeline**: Chains Workflows 1 → 1.5 → 2 → 3 autonomously. |
| **research-refine-pipeline** | `/research-refine-pipeline [idea]` | Chains `research-refine → experiment-plan → research-review` for proposal refinement. |
| **paper-writing** | `/paper-writing [report]` | **Workflow 3**: Chains `paper-plan → paper-figure → paper-write → paper-compile → auto-improvement`. |

### Notifications

| Skill | Invoke | Description |
|-------|--------|-------------|
| **feishu-notify** | `/feishu-notify [message]` | Send notifications to Feishu/Lark. Supports experiment status, review scores, checkpoints. |

### Cross-Platform

| Skill | Invoke | Description |
|-------|--------|-------------|
| **arxiv** | `/arxiv [query]` | Search and fetch papers from arXiv. Structured output with abstracts and metadata. |
| **grant-proposal** | `/grant-proposal [idea]` | Draft structured grant proposals. 9 agencies: KAKENHI, NSF, NSFC (6 subtypes), ERC, DFG, SNSF, ARC, NWO. |

---

## MCP Servers

ARIS includes 4 MCP servers for cross-model collaboration:

| Server | Purpose | Setup |
|--------|---------|-------|
| **claude-review** | Codex executes, Claude reviews via local bridge | `claude mcp add claude-review` |
| **feishu-bridge** | Feishu/Lark webhook notifications | Configure `FEISHU_WEBHOOK_URL` |
| **llm-chat** | Generic OpenAI-compatible LLM as reviewer | Configure `LLM_API_KEY`, `LLM_BASE_URL` |
| **minimax-chat** | MiniMax model as reviewer | Configure `MINIMAX_API_KEY` |

---

## Codex CLI Skills

All 31 core skills are also available as native [Codex CLI](https://github.com/openai/codex) skills in `skills/skills-codex/`, using `spawn_agent` instead of Claude Code subagents.

## Adding a New Skill

1. Create a directory under `aris/skills/<skill-name>/`
2. Add a `SKILL.md` file with YAML frontmatter:
   ```yaml
   ---
   name: my-skill
   description: "Brief description of what the skill does"
   argument-hint: [required-args]
   allowed-tools: Bash(*), Read, Write, Edit, ...
   ---
   ```
3. Write the skill body as Markdown instructions
4. Register the skill by copying to `~/.claude/skills/` or adding to your project's `.claude/skills/`
