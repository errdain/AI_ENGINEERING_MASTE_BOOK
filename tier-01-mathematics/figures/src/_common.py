import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures-common"))
sys.path.insert(0, str(ROOT / "tier-01-mathematics" / "code"))
OUT = pathlib.Path(__file__).resolve().parents[1]
import figstyle
figstyle.apply()
