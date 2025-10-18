#!/usr/bin/env python3

from pathlib import Path
import argparse, json, sys, pandas as pd

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Analyse DeepCubeA test logs and print per-agent and ensemble statistics.")
parser.add_argument("A",      type=int, nargs="?", default=2,
                    help="how many last model_ids to analyse (default: 2)")
parser.add_argument("EPOCH",  nargs="?", default="128",
                    help="epoch used in the log file names (default: 128)")
parser.add_argument("B",      nargs="?", default="1048576",
                    help="value of B in the log file names (default: 1048576)")
args = parser.parse_args()

A, EPOCH, B = args.A, args.EPOCH, args.B
DATASET     = "deepcubea"
PREFIX      = "test_p054-t000"
LOGS        = Path("logs")

# ---------- read last A model_ids ----------
mid_path = LOGS / "model_id.txt"
model_ids = mid_path.read_text().splitlines()[-A:]
if not model_ids:
    sys.exit("model_id.txt is empty")

# ---------- collect rows ----------
rows = []
for mid in model_ids:
    file = LOGS / f"{PREFIX}-{DATASET}_{mid}_{EPOCH}_B{B}.json"
    if not file.exists():
        print(f"warning: {file.name} not found, skipped", file=sys.stderr)
        continue
    with open(file) as fh:
        for rec in json.load(fh):
            rows.append({
                "test_num":        rec["test_num"],
                "solution_length": rec["solution_length"],
                "moves":           rec["moves"],
                "model_id":        mid,
            })

if not rows:
    sys.exit("no matching test_*.json files found")

df = pd.DataFrame(rows)
df["solution_length"] = pd.to_numeric(df["solution_length"], errors="coerce")

# ---------- per-agent statistics ----------
g = df.groupby("model_id")
per_agent = pd.DataFrame({
    "tests"   : g.size(),
    "solved_%": (100 * g["solution_length"].apply(lambda s: s.notna().mean())).round(1),
    "avg_len" : g["solution_length"].mean().round(2),
}).sort_values("avg_len")

print("\n=== per agent ===")
print(per_agent.to_string())

# ---------- ensemble statistics ----------
best = df.groupby("test_num")["solution_length"].min()
print("\n=== ensemble (shortest per scramble) ===")
print(f"solved %           : {100 * best.notna().mean():.1f}")
print(f"avg solution length: {best.mean():.2f}")

# ---------- moves of best  agent ----------
winners = df.loc[df.groupby("test_num")["solution_length"].idxmin()] \
           .sort_values("test_num")
pd.set_option("display.max_colwidth", None)
print("\n")
print(winners[["test_num", "solution_length", "model_id", "moves"]].to_string(index=False))