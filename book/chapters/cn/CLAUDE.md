# Chinese Translation — Sync Workflow

This directory contains the Chinese translation of all book chapters.
Each file mirrors its English counterpart in `book/chapters/` and begins
with `<!-- source-commit: <hash> -->` recording the last-synced commit.

## How to sync a file after the original changes

### 1. Find what changed

```bash
# Get the hash recorded in the translated file
SOURCE_COMMIT=$(head -1 book/chapters/cn/01-introduction.md | grep -o '[0-9a-f]\{7,40\}')

# See the diff since that commit
git diff $SOURCE_COMMIT HEAD -- book/chapters/01-introduction.md
```

To check all files at once:

```bash
for f in book/chapters/cn/*.md; do
  base=$(basename "$f")
  original="book/chapters/$base"
  [ -f "$original" ] || continue
  hash=$(head -1 "$f" | grep -o '[0-9a-f]\{7,40\}')
  [ -z "$hash" ] && continue
  count=$(git diff --shortstat "$hash" HEAD -- "$original" 2>/dev/null | grep -o '[0-9]* insertion' | cut -d' ' -f1)
  [ -n "$count" ] && echo "$base: ~$count lines changed since $hash"
done
```

### 2. Translate only the changed sections

Read the diff, identify which paragraphs changed, and update just those
sections in the translated file. Do **not** re-translate the whole file.

Use the chunked agent approach (120 lines per agent) if sections are large.
Agents that try to translate an entire large file in one pass will time out.

### 3. Update the source-commit hash

After updating a file, change the first line to the current HEAD:

```bash
CURRENT=$(git rev-parse --short HEAD)
# Replace the source-commit line at the top of the translated file
sed -i "1s/.*/<!-- source-commit: $CURRENT -->/" book/chapters/cn/01-introduction.md
```

---

## Translation guidelines

**Keep in English** (do not translate):
- Technical terms: RLHF, SFT, RL, PPO, DPO, GRPO, RLOO, LLM, IFT, PreFT,
  RLVR, reward model, policy, policy gradient, value function, KL divergence,
  fine-tuning, token, dataset, benchmark, prompt, completion, rollout,
  trajectory, advantage function, actor-critic, chain-of-thought, inference,
  post-training, Constitutional AI, RLAIF, etc.
- Proper nouns: ChatGPT, GPT-4, OpenAI, Anthropic, Claude, Llama, DeepSeek,
  Tülu, InstructGPT, Zephyr, HuggingFace, Ai2, etc.
- Citation keys: `[@rafailov2024direct]` — unchanged verbatim
- Figure/equation/table references: `@fig:rlhf-basic`, `{#fig:...}`, `@eq:...`
- Image paths: `![...](images/filename.png)`
- Math/LaTeX: all `$...$` and `$$...$$` blocks
- Code blocks: all fenced code blocks unchanged
- HTML comments: `<!-- ... -->`
- YAML front matter **keys**: `prev-chapter:`, `page-title:`, etc.
- URLs in front matter: `prev-url:`, `next-url:`, `lecture-video:`

**Translate**:
- All prose paragraphs
- Section headings (`#`, `##`, `###`)
- YAML front matter **values** that are human-readable labels:
  - `page-title: Introduction` → `page-title: 引言`
  - `prev-chapter: "Home"` → `prev-chapter: "主页"`
  - `lecture-label: "Lecture 1: ..."` → `lecture-label: "第一讲：..."`
- List items and table cells that contain prose
- Figure captions (the `![Caption](...)` alt text)

---

## Chunked agent approach (avoids timeouts)

For files larger than ~150 lines, use one agent per ~120-line chunk.
Each agent reads its assigned range with `offset` and `limit`, translates,
and writes to a numbered temp file. Assemble with `cat` afterwards.

**Chunk 1** (includes front matter):
```
Read offset=0 limit=120
Output: cn/tmp/<filename>.chunk01.md
Include: <!-- source-commit: HASH --> as line 1, then copyright comment,
         then translated YAML front matter, then translated content.
```

**Chunks 2+** (prose only):
```
Read offset=120 limit=120   (adjust for each chunk)
Output: cn/tmp/<filename>.chunk02.md
No header, no front matter — just the translated prose.
```

**Assemble**:
```bash
cat cn/tmp/06-policy-gradients.chunk{01..11}.md > cn/06-policy-gradients.md
```

Chapter sizes for reference:
- 06-policy-gradients.md: 1322 lines → 11 chunks
- 05-reward-models.md: 521 lines → 4 chunks
- Most others: 150–370 lines → 1–3 chunks
