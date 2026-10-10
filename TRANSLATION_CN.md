# Chinese Translation Status

Translated files are in `book/chapters/cn/`. Each file's first line records the upstream
commit (`natolambert/rlhf-book`) whose English text it was last synced to. Last full sync:
`eecc49e` (upstream `main`, merged 2026-10-10). `appendix-00-references` has no upstream
changes since `02c6d49`, so it keeps that hash.

## How to Sync Future Changes

See `book/chapters/cn/CLAUDE.md` for the full workflow. In short:

```bash
git fetch upstream main && git merge upstream/main

# See what changed in a specific file since its recorded translation commit
git diff 02c6d49 HEAD -- book/chapters/01-introduction.md   # use the hash from the CN file's line 1

# Check a translated file's recorded source commit
head -1 book/chapters/cn/01-introduction.md
# Output: <!-- source-commit: eecc49e -->
```

After updating a translation, set the first line of the translated file to the new upstream commit hash.

## File Status

| Original | Translation | Source Commit | Status |
|---|---|---|---|
| book/chapters/01-introduction.md | book/chapters/cn/01-introduction.md | eecc49e | ✓ Translated |
| book/chapters/02-related-works.md | book/chapters/cn/02-related-works.md | eecc49e | ✓ Translated |
| book/chapters/03-training-overview.md | book/chapters/cn/03-training-overview.md | eecc49e | ✓ Translated |
| book/chapters/04-instruction-tuning.md | book/chapters/cn/04-instruction-tuning.md | eecc49e | ✓ Translated |
| book/chapters/05-reward-models.md | book/chapters/cn/05-reward-models.md | eecc49e | ✓ Translated |
| book/chapters/06-policy-gradients.md | book/chapters/cn/06-policy-gradients.md | eecc49e | ✓ Translated |
| book/chapters/07-reasoning.md | book/chapters/cn/07-reasoning.md | eecc49e | ✓ Translated |
| book/chapters/08-direct-alignment.md | book/chapters/cn/08-direct-alignment.md | eecc49e | ✓ Translated |
| book/chapters/09-rejection-sampling.md | book/chapters/cn/09-rejection-sampling.md | eecc49e | ✓ Translated |
| book/chapters/10-preferences.md | book/chapters/cn/10-preferences.md | eecc49e | ✓ Translated |
| book/chapters/11-preference-data.md | book/chapters/cn/11-preference-data.md | eecc49e | ✓ Translated |
| book/chapters/12-synthetic-data.md | book/chapters/cn/12-synthetic-data.md | eecc49e | ✓ Translated |
| book/chapters/13-tools.md | book/chapters/cn/13-tools.md | eecc49e | ✓ Translated |
| book/chapters/14-over-optimization.md | book/chapters/cn/14-over-optimization.md | eecc49e | ✓ Translated |
| book/chapters/15-regularization.md | book/chapters/cn/15-regularization.md | eecc49e | ✓ Translated |
| book/chapters/16-evaluation.md | book/chapters/cn/16-evaluation.md | eecc49e | ✓ Translated |
| book/chapters/17-product.md | book/chapters/cn/17-product.md | eecc49e | ✓ Translated |
| book/chapters/appendix-00-references.md | book/chapters/cn/appendix-00-references.md | 02c6d49 | ✓ Translated |
| book/chapters/appendix-a-definitions.md | book/chapters/cn/appendix-a-definitions.md | eecc49e | ✓ Translated |
| book/chapters/appendix-b-style.md | book/chapters/cn/appendix-b-style.md | eecc49e | ✓ Translated |
| book/chapters/appendix-c-practical.md | book/chapters/cn/appendix-c-practical.md | eecc49e | ✓ Translated |

## Translation Guidelines

- **Keep in English**: all technical terms (RLHF, SFT, RL, PPO, DPO, LLM, reward model, policy, fine-tuning, token, etc.), proper nouns (ChatGPT, OpenAI, Anthropic, etc.), citation keys `[@...]`, figure references, math/LaTeX, code blocks
- **Translate**: all prose, section headings, YAML front matter human-readable values
