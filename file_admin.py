import json
from pathlib import Path
import pandas as pd
import numpy as np

for bonus in [0,5]:
    for seed in np.arange(1,10):
        for nf in [0,2,4,8,16,32]:
            results_dir = Path(f"/Users/alecsargood/Desktop/Tractography/Results/Fibercup/nf_seed{seed}_bonus{bonus}/{nf}/validate/tractometer/scores")
            with open(results_dir / f"tractogram_nf_seed{seed}_bonus{bonus}_{nf}_fibercup_3mm.json","r") as f:
                data = json.load(f)
            VC = data["VC"]
            VB = data["VB"]
            df = pd.concat([df, 
                            pd.DataFrame({
                "bonus":[bonus],
                "seed":seed,
                "num_flows":nf,
                "VC":VC,
                "VB":VB
            })], ignore_index=True)