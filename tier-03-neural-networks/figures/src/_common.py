import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures-common"))
sys.path.insert(0, str(ROOT / "tier-03-neural-networks" / "code"))
sys.path.insert(0, str(ROOT / "tier-03-neural-networks" / "lab"))
OUT = pathlib.Path(__file__).resolve().parents[1]
import figstyle
figstyle.apply()
