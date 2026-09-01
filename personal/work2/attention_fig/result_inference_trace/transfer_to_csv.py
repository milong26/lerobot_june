import torch
import pandas as pd

def analyze_trace(path, model_name):
    trace = torch.load(path, map_location="cpu")

    rows = []

    for i, v in enumerate(trace["v_t"]):
        rows.append({
            "model": model_name,
            "step": i,
            "v_t_norm": v.norm().item()
        })

    return pd.DataFrame(rows)


df1 = analyze_trace(
    "random_corner_16k/seed_10042/trace.pt",
    "random"
)

df2 = analyze_trace(
    "ours_corner_16k/seed_10042/trace.pt",
    "ours"
)
df1 = analyze_trace(
    "random_corner_16k/seed_10042/trace.pt",
    "random"
)

df3 = analyze_trace(
    "uniform_corner_16k/seed_10042/trace.pt",
    "uniform"
)
df = pd.concat([df1, df2, df3])

df.to_csv(
    "v_t_norm_analysis.csv",
    index=False
)