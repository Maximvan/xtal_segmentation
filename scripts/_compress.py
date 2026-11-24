import sys
from pathlib import Path
import numpy as np
from xtal_lib import save
for f in sorted(Path(sys.argv[1]).glob("*.npy")):
    if f.name == "bbox.npy":
        continue
    a = np.load(f); save(f, a); print(f.name, "->", round(f.with_suffix(".npz").stat().st_size / 1e6, 1), "MB")
