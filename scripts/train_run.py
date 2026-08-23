import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app

start = time.time()
app.training_done.wait()
print(f"TRAINING COMPLETE in {time.time()-start:.0f}s")
