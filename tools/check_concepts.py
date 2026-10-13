"""Validate a tier folder: one file per roadmap concept, exact 23 headings in order, no broken relative links/images.
usage: python tools/check_concepts.py <tier-folder>   (folder name tier-NN-..., concept files directly inside it)"""
import re, sys, subprocess, json, pathlib
folder = pathlib.Path(sys.argv[1]); tier = int(re.search(r"tier-(\d+)", folder.name).group(1))
exp = json.loads(subprocess.check_output([sys.executable, str(pathlib.Path(__file__).parent / "concept_list.py"), str(tier), "--json"]))
H = ["1. WHAT IS IT?","2. WHY DOES IT EXIST?","3. DETAILED EXPLANATION","4. HOW DOES IT WORK?","5. SIMPLE EXAMPLE","6. TECHNICAL EXAMPLE","7. CODE","8. INPUT → PROCESS → OUTPUT","9. ENTERPRISE USE CASE","10. AI ENGINEERING CONNECTION","11. ADVANTAGES","12. LIMITATIONS","13. TRADE-OFFS","14. COMMON MISTAKES","15. PERFORMANCE / COST CONSIDERATIONS","16. SECURITY CONSIDERATIONS","17. INTERVIEW EXPLANATION","18. RAPID-FIRE ANSWER","19. INTERVIEW QUESTIONS","20. HANDS-ON EXERCISE","21. REVISION QUESTIONS","22. RELATED CONCEPTS","23. ONE-LINE MEMORY HOOK"]
errs = []
have = {p.name for p in folder.glob("[0-9][0-9].[0-9][0-9]-*.md")}
for num, title, path in exp:
    name = path.split("/")[-1]
    if name not in have: errs.append(f"missing {name}"); continue
    t = (folder / path).read_text(encoding="utf-8")
    if not re.search(rf"^## {re.escape(num)} ", t, re.M): errs.append(f"{name}: missing '## {num} ' title")
    if "Metadata" not in t: errs.append(f"{name}: no Metadata")
    got = re.findall(r"^### (\d+\. .*)$", t, re.M)
    if [g.strip() for g in got] != H: errs.append(f"{name}: headings differ ({len(got)} found)")
extra = have - {p.split('/')[-1] for _, _, p in exp}
for e in sorted(extra): errs.append(f"unexpected file {e}")
for md in list(folder.rglob("*.md")):
    for m in re.finditer(r"!?\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", md.read_text(encoding="utf-8")):
        l = m.group(1)
        if re.match(r"(https?:|mailto:)", l): continue
        if not (md.parent / l).resolve().exists(): errs.append(f"{md.relative_to(folder)}: broken link {l}")
print(f"{len(exp)} concepts expected, {len(have)} present"); [print("ERR", e) for e in errs[:60]]
print("OK" if not errs else f"{len(errs)} problems"); sys.exit(1 if errs else 0)
