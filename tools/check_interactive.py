"""Smoke-test interactive HTML files: load in headless Chromium, fail on JS errors,
wiggle every slider/checkbox, save a screenshot.
usage: python tools/check_interactive.py <dir-or-html>... [--shots OUTDIR]"""
import sys, pathlib
from playwright.sync_api import sync_playwright

args = [a for a in sys.argv[1:] if not a.startswith("--")]
shots = pathlib.Path(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else None
if shots:
    args.remove(str(shots)); shots.mkdir(parents=True, exist_ok=True)
files = []
for a in args:
    p = pathlib.Path(a)
    files += sorted(p.rglob("*.html")) if p.is_dir() else [p]
bad = 0
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium" if pathlib.Path("/opt/pw-browsers/chromium").exists() else None)
    for f in files:
        pg = b.new_page(viewport={"width": 1000, "height": 900}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(f"file://{f.resolve()}")
        for el in pg.query_selector_all("input[type=range]"):
            mn, mx = el.get_attribute("min") or "0", el.get_attribute("max") or "100"
            for v in (mn, mx, str((float(mn) + float(mx)) / 2)):
                el.evaluate("(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}))}", v)
        for el in pg.query_selector_all("input[type=checkbox]"):
            el.click(); el.click()
        for el in pg.query_selector_all("button"):
            try: el.click(timeout=500)
            except Exception: pass
        pg.wait_for_timeout(200)
        if shots: pg.screenshot(path=str(shots / (f.stem + ".png")), full_page=True)
        print(("FAIL " if errs else "ok   ") + str(f), errs[:3]); bad += bool(errs); pg.close()
    b.close()
sys.exit(1 if bad else 0)
