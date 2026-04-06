# Chinese Translation Status

Translated files are in `book/chapters/cn/`. All chapters translated from source commit `02c6d49`.

## How to Sync Future Changes

When the original English chapters change, find what needs updating:

```bash
# See what changed in a specific file since translation
git diff 02c6d49 HEAD -- book/chapters/01-introduction.md

# See all changed chapter files since translation
git diff --name-only 02c6d49 HEAD -- book/chapters/

# Check if a translated file is stale (has source-commit older than HEAD)
head -1 book/chapters/cn/01-introduction.md
# Output: <!-- source-commit: 02c6d49 -->
```

After updating a translation, update the source-commit comment at the top of the translated file to the new commit hash.

## File Status

| Original | Translation | Source Commit | Status |
|---|---|---|---|
| book/chapters/01-introduction.md | book/chapters/cn/01-introduction.md | 02c6d49 | ✓ Translated |
| book/chapters/02-related-works.md | book/chapters/cn/02-related-works.md | 02c6d49 | ✓ Translated |
| book/chapters/03-training-overview.md | book/chapters/cn/03-training-overview.md | 02c6d49 | ✓ Translated |
| book/chapters/04-instruction-tuning.md | book/chapters/cn/04-instruction-tuning.md | 02c6d49 | ✓ Translated |
| book/chapters/05-reward-models.md | book/chapters/cn/05-reward-models.md | 02c6d49 | ✓ Translated |
| book/chapters/06-policy-gradients.md | book/chapters/cn/06-policy-gradients.md | 02c6d49 | ✓ Translated |
| book/chapters/07-reasoning.md | book/chapters/cn/07-reasoning.md | 02c6d49 | ✓ Translated |
| book/chapters/08-direct-alignment.md | book/chapters/cn/08-direct-alignment.md | 02c6d49 | ✓ Translated |
| book/chapters/09-rejection-sampling.md | book/chapters/cn/09-rejection-sampling.md | 02c6d49 | ✓ Translated |
| book/chapters/10-preferences.md | book/chapters/cn/10-preferences.md | 02c6d49 | ✓ Translated |
| book/chapters/11-preference-data.md | book/chapters/cn/11-preference-data.md | 02c6d49 | ✓ Translated |
| book/chapters/12-synthetic-data.md | book/chapters/cn/12-synthetic-data.md | 02c6d49 | ✓ Translated |
| book/chapters/13-tools.md | book/chapters/cn/13-tools.md | 02c6d49 | ✓ Translated |
| book/chapters/14-over-optimization.md | book/chapters/cn/14-over-optimization.md | 02c6d49 | ✓ Translated |
| book/chapters/15-regularization.md | book/chapters/cn/15-regularization.md | 02c6d49 | ✓ Translated |
| book/chapters/16-evaluation.md | book/chapters/cn/16-evaluation.md | 02c6d49 | ✓ Translated |
| book/chapters/17-product.md | book/chapters/cn/17-product.md | 02c6d49 | ✓ Translated |
| book/chapters/appendix-00-references.md | book/chapters/cn/appendix-00-references.md | 02c6d49 | ✓ Translated |
| book/chapters/appendix-a-definitions.md | book/chapters/cn/appendix-a-definitions.md | 02c6d49 | ✓ Translated |
| book/chapters/appendix-b-style.md | book/chapters/cn/appendix-b-style.md | 02c6d49 | ✓ Translated |
| book/chapters/appendix-c-practical.md | book/chapters/cn/appendix-c-practical.md | 02c6d49 | ✓ Translated |
| book/chapters/README.md | book/chapters/cn/README.md | 02c6d49 | ✓ Translated |

## Translation Guidelines

- **Keep in English**: all technical terms (RLHF, SFT, RL, PPO, DPO, LLM, reward model, policy, fine-tuning, token, etc.), proper nouns (ChatGPT, OpenAI, Anthropic, etc.), citation keys `[@...]`, figure references, math/LaTeX, code blocks
- **Translate**: all prose, section headings, YAML front matter human-readable values
