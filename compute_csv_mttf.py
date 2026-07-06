import pandas as pd
import glob
import os
import numpy as np
import math

files = glob.glob('results/*_results.csv')
for f in sorted(files):
    df = pd.read_csv(f)
    if 'run' in df.columns:
        df = df.drop(columns=['run'])
    # sum along axis 1 gives ttf for each run
    ttfs = df.sum(axis=1)
    mttf = np.mean(ttfs)
    std = np.std(ttfs, ddof=1)
    n = len(ttfs)
    se = std / math.sqrt(n)
    ci_low = mttf - 1.96 * se
    ci_high = mttf + 1.96 * se
    name = os.path.basename(f)
    print(f"{name}: MTTF = {mttf:.4f} \\pm {se:.4f}, 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
