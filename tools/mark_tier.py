"""usage: mark_tier.py NN "Topic" "folder-slug" "README title" -- adds/updates rows in CHAPTER_STATUS.md and phase README, sorted."""
import sys, re
nn, topic, slug, title = sys.argv[1:5]
P = "phase-2-transformers-foundation-models/README.md"
row = f"| {nn} | [{title}]({slug}/README.md) | AUTHOR_DRAFT |"
lines = open(P).read().rstrip("\n").split("\n")
lines = [l for l in lines if not l.startswith(f"| {nn} |")]
idx = [i for i, l in enumerate(lines) if re.match(r"\| \d\d \|", l)]
pos = next((i for i in idx if lines[i][2:4] > nn), idx[-1] + 1)
lines.insert(pos, row); open(P, "w").write("\n".join(lines) + "\n")
S = "CHAPTER_STATUS.md"
lines = open(S).read().rstrip("\n").split("\n")
lines = [l for l in lines if not l.startswith(f"| {nn} |")]
new = f"| {nn} | {topic} | Claude | ChatGPT | Codex | ChatGPT | AUTHOR_DRAFT (code TESTED) |"
idx = [i for i, l in enumerate(lines) if re.match(r"\| \d\d \|", l)]
pos = next((i for i in idx if lines[i][2:4] > nn), idx[-1] + 1)
lines.insert(pos, new); open(S, "w").write("\n".join(lines) + "\n")
