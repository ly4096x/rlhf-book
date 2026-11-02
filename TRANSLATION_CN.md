# Chinese Translation Status

Translated files are in `book/chapters/cn/`. Each file's first line records the upstream
commit (`natolambert/rlhf-book`) whose English text it was last synced to. Last full sync:
`0106c01` (upstream `main`, merged 2026-11-02). `appendix-00-references` has no upstream
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
| book/chapters/01-introduction.md | book/chapters/cn/01-introduction.md | 0106c01 | ✓ Translated |
| book/chapters/02-related-works.md | book/chapters/cn/02-related-works.md | eecc49e | ✓ Translated |
| book/chapters/03-training-overview.md | book/chapters/cn/03-training-overview.md | 0106c01 | ✓ Translated |
| book/chapters/04-instruction-tuning.md | book/chapters/cn/04-instruction-tuning.md | eecc49e | ✓ Translated |
| book/chapters/05-reward-models.md | book/chapters/cn/05-reward-models.md | 0106c01 | ✓ Translated |
| book/chapters/06-policy-gradients.md | book/chapters/cn/06-policy-gradients.md | 0106c01 | ✓ Translated |
| book/chapters/07-reasoning.md | book/chapters/cn/07-reasoning.md | 0106c01 | ✓ Translated |
| book/chapters/08-direct-alignment.md | book/chapters/cn/08-direct-alignment.md | 0106c01 | ✓ Translated |
| book/chapters/09-rejection-sampling.md | book/chapters/cn/09-rejection-sampling.md | eecc49e | ✓ Translated |
| book/chapters/10-preferences.md | book/chapters/cn/10-preferences.md | eecc49e | ✓ Translated |
| book/chapters/11-preference-data.md | book/chapters/cn/11-preference-data.md | 0106c01 | ✓ Translated |
| book/chapters/12-synthetic-data.md | book/chapters/cn/12-synthetic-data.md | eecc49e | ✓ Translated |
| book/chapters/13-tools.md | book/chapters/cn/13-tools.md | 0106c01 | ✓ Translated |
| book/chapters/14-over-optimization.md | book/chapters/cn/14-over-optimization.md | eecc49e | ✓ Translated |
| book/chapters/15-regularization.md | book/chapters/cn/15-regularization.md | 0106c01 | ✓ Translated |
| book/chapters/16-evaluation.md | book/chapters/cn/16-evaluation.md | eecc49e | ✓ Translated |
| book/chapters/17-product.md | book/chapters/cn/17-product.md | 0106c01 | ✓ Translated |
| book/chapters/appendix-00-references.md | book/chapters/cn/appendix-00-references.md | 02c6d49 | ✓ Translated |
| book/chapters/appendix-a-definitions.md | book/chapters/cn/appendix-a-definitions.md | 0106c01 | ✓ Translated |
| book/chapters/appendix-b-style.md | book/chapters/cn/appendix-b-style.md | eecc49e | ✓ Translated |
| book/chapters/appendix-c-practical.md | book/chapters/cn/appendix-c-practical.md | 0106c01 | ✓ Translated |

## Other translated resources

| Original | Translation | Source Commit |
|---|---|---|
| book/templates/course.html | book/cn/course.html | 0106c01 |
| book/templates/library.html | book/cn/library.html | 0106c01 |
| book/rl-cheatsheet/index.html | book/cn/rl-cheatsheet.html | 0106c01 |
| book/rl-cheatsheet/inside_cover_back.tex | book/cn/rl-cheatsheet.tex | 0106c01 |
| book/metadata.yml | book/metadata-cn.yml | 0106c01 |

Slide decks (built by `make cn-teach`, see `book/chapters/cn/CLAUDE.md`):

| Original | Translation | Source Commit |
|---|---|---|
| teach/course/lec0-prereq-review.md | teach/cn/course/lec0-prereq-review.md | 0106c01 |
| teach/course/lec1-chap1-3.md | teach/cn/course/lec1-chap1-3.md | 0106c01 |
| teach/course/lec2-chap4-5-9.md | teach/cn/course/lec2-chap4-5-9.md | 0106c01 |
| teach/course/lec3-chap6-p1.md | teach/cn/course/lec3-chap6-p1.md | 0106c01 |
| teach/course/lec4-chap6-p2.md | teach/cn/course/lec4-chap6-p2.md | 0106c01 |
| teach/course/lec5-chap7.md | teach/cn/course/lec5-chap7.md | 0106c01 |
| teach/course/lec6-chap8-dpo.md | teach/cn/course/lec6-chap8-dpo.md | 0106c01 |
| teach/course/lec7-chap12-synthetic-data.md | teach/cn/course/lec7-chap12-synthetic-data.md | 0106c01 |
| teach/course/lec8-chap10-11-preferences.md | teach/cn/course/lec8-chap10-11-preferences.md | 0106c01 |
| teach/course/lec9-chap14-appb-overoptimization.md | teach/cn/course/lec9-chap14-appb-overoptimization.md | 0106c01 |
| teach/course/lec10-chap15-regularization.md | teach/cn/course/lec10-chap15-regularization.md | 0106c01 |
| teach/course/lec11-chap13-tools.md | teach/cn/course/lec11-chap13-tools.md | 0106c01 |
| teach/course/lec12-chap16-evals.md | teach/cn/course/lec12-chap16-evals.md | 0106c01 |
| teach/course/lec13-chap17-character.md | teach/cn/course/lec13-chap17-character.md | 0106c01 |
| teach/course/qa-01.md | teach/cn/course/qa-01.md | 0106c01 |
| teach/course/qa-02.md | teach/cn/course/qa-02.md | 0106c01 |
| teach/course/qa-03.md | teach/cn/course/qa-03.md | 0106c01 |
| teach/course/conversation-01.md | teach/cn/course/conversation-01.md | 0106c01 |
| teach/extras/01-shoggoths.md | teach/cn/extras/01-shoggoths.md | 0106c01 |
| teach/SALA-2026/talk.md | teach/cn/SALA-2026/talk.md | 0106c01 |

Not translated: `teach/assets/from-research-to-practice.pdf` (produced outside this repo), text inside slide images, and the YouTube recordings.

## Translation Guidelines

- **Keep in English**: all technical terms (RLHF, SFT, RL, PPO, DPO, LLM, reward model, policy, fine-tuning, token, etc.), proper nouns (ChatGPT, OpenAI, Anthropic, etc.), citation keys `[@...]`, figure references, math/LaTeX, code blocks
- **Translate**: all prose, section headings, YAML front matter human-readable values
