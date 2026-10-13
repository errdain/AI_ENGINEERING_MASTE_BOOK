"""Print canonical concept list for a tier from the frozen roadmap.
usage: python tools/concept_list.py 6          -> lines: "6.1<TAB>Text preprocessing<TAB>06.01-text-preprocessing.md"
       python tools/concept_list.py 6 --json
Filenames are canonical: {tier:02d}.{n:02d}-{slug}.md (inside tier-NN-<slug>/)"""
import re, sys, json
tier = int(sys.argv[1])
text = open(__file__.rsplit("/tools/", 1)[0] + "/ROADMAP_V1.0_FROZEN.md", encoding="utf-8").read()
m = re.search(rf"^Tier {tier} --- .*?$", text, re.M)
start = m.end()
nxt = re.search(r"^(Tier \d+ --- |PHASE \d+ ---)", text[start:], re.M)
body = text[start:start + nxt.start()] if nxt else text[start:]
body = re.split(r"\n(?:Build|Major build|Rule|Tier \d+ also)", body)[0]
body = re.sub(r"\s+", " ", body)
parts = re.split(rf"(?:(?<=\s)|^){tier}\.(\d+) ", " " + body)
items = []
for i in range(1, len(parts), 2):
    items.append((int(parts[i]), parts[i + 1].strip(" ⸻")))
def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]
out = [(f"{tier}.{n}", t, f"{tier:02d}.{n:02d}-{slug(t)}.md") for n, t in items]
if "--json" in sys.argv: print(json.dumps(out))
else:
    for o in out: print("\t".join(o))
