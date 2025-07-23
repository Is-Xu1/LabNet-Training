import os
import numpy as np
import pandas as pd
import torch
from obspy import read
from seisbench.models import VariableLengthPhaseNet
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Config ---
WINDOW_SIZE = 50000
STRIDE = 5000
SAMPLING_RATE = 5000000
THRESHOLD = 0.5
TOLERANCE = 50
CHECKPOINT_PATH = "50000w100gANR_2.pt"
DATA_DIR = "f:/Data"
VALIDATION_CSV = "p_picks_validation.csv"
OUTPUT_CSV = f"{CHECKPOINT_PATH}.csv"

# --- CUDA support ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📟 Using device: {DEVICE}")

# --- Load model ---
model = VariableLengthPhaseNet(
    in_channels=1,
    classes=2,
    phases="NP",
    sampling_rate=SAMPLING_RATE,
    norm="std"
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

# --- Helper functions ---
def parse_name(name):
    parts = name.split("_")
    trace_index = int(parts[-1].replace("trace", "")) - 1
    event = f"{parts[-3]}_{parts[-2]}"
    exp = f"{parts[2]}_{parts[3]}"
    run_parts = parts[4:-3]
    run = "_".join(run_parts)
    return exp, run, event, trace_index

def load_waveform(name_key):
    exp, run, event, trace_index = parse_name(name_key)
    base_folder = os.path.join(DATA_DIR, exp)
    folder = os.path.join(base_folder, run)
    if not os.path.exists(folder):
        alt_run = run.replace("_Traces", "") if "_Traces" in run else run + "_Traces"
        folder = os.path.join(base_folder, alt_run)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"❌ Neither '{run}' nor fallback '{alt_run}' exists under {base_folder}")
    filename = f"{event}_WindowSize_0.05s_Data.mseed"
    full_path = os.path.join(folder, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File not found: {full_path}")
    stream = read(full_path)
    return stream[trace_index].data.astype(np.float32), full_path

# --- Inference and CSV generation ---
df = pd.read_csv(VALIDATION_CSV)
results = []
prediction_cache = {}

# For stats
residuals_under_1000 = []
residuals_over_1000 = 0
all_recorded_residuals = []
no_predicted_picks = 0

for _, row in df.iterrows():
    name = row["Name"]

    # --- Support multiple picks (single int or list) ---
    raw_pick = row["marked_point"]
    if isinstance(raw_pick, str) and raw_pick.strip().startswith('['):
        pick_list = eval(raw_pick)
    else:
        try:
            val = float(raw_pick)
            pick_list = [int(val)] if val >= 0 else []
        except:
            pick_list = []
    has_pick = len(pick_list) > 0

    try:
        waveform, _ = load_waveform(name)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        continue

    original_length = len(waveform)

    # --- Zero-pad short waveforms ---
    if original_length < WINDOW_SIZE:
        padded_waveform = np.zeros(WINDOW_SIZE, dtype=np.float32)
        padded_waveform[:original_length] = waveform
        waveform = padded_waveform
    else:
        padded_waveform = waveform.copy()

    # --- Normalize entire waveform once ---
    std = waveform.std()
    if std > 1e-6:
        waveform = (waveform - waveform.mean()) / (std + 1e-6)
    else:
        waveform = np.zeros_like(waveform)

    probs = np.zeros((2, len(waveform)))
    count = np.zeros(len(waveform))

    # --- Sliding window inference ---
    for start in range(0, len(waveform) - WINDOW_SIZE + 1, STRIDE):
        segment = waveform[start:start + WINDOW_SIZE]
        input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)[0].detach().cpu().numpy()
        probs[:, start:start + WINDOW_SIZE] += output
        count[start:start + WINDOW_SIZE] += 1

    last_start = len(waveform) - WINDOW_SIZE
    if last_start % STRIDE != 0:
        remaining = waveform[last_start:]
        padded = np.zeros(WINDOW_SIZE, dtype=np.float32)
        padded[:len(remaining)] = remaining
        input_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)[0].detach().cpu().numpy()
        probs[:, last_start:] += output[:, :len(remaining)]
        count[last_start:] += 1

    count[count == 0] = 1
    probs /= count

    probs_p = probs[1]
    probs_noise = probs[0]

    # --- Peak detection ---
    peak_indices, _ = find_peaks(probs_p, height=THRESHOLD, prominence=0.3, distance=3000)
    tp, fp, fn = 0, 0, 0
    residual = None

    # --- Residual tracking, now per ground-truth pick (skip infinities) ---
    residuals_list = []
    for gt in pick_list:
        if len(peak_indices) > 0:
            dists = [abs(p - gt) for p in peak_indices]
            min_dist = min(dists)
        else:
            min_dist = float('inf')

        # only record real (finite) residuals
        if np.isfinite(min_dist):
            all_recorded_residuals.append(min_dist)
            residuals_list.append(min_dist)
            if min_dist <= 1000:
                residuals_under_1000.append(min_dist)
            else:
                residuals_over_1000 += 1
        # if min_dist is inf, we simply skip it here

    # store the per-pick finite residuals
    residual = residuals_list


    # --- Confusion counts (unchanged) ---
    if len(peak_indices) == 0:
        no_predicted_picks += 1
        if has_pick:
            fn = len(pick_list)
        else:
            tp = 1
    else:
        matched_gts = set()
        for p in peak_indices:
            if has_pick:
                close = [gt for gt in pick_list if abs(p - gt) <= TOLERANCE and gt not in matched_gts]
                if close:
                    matched_gts.add(close[0])
                    tp += 1
                    continue
                if any(abs(p - gt) <= 1000 for gt in pick_list):
                    continue
                fp += 1
            else:
                fp += 1
        if has_pick:
            fn = len(pick_list) - len(matched_gts)

    results.append({
        "Name": name,
        "True Pick": pick_list,
        "Predicted Picks": peak_indices.tolist(),
        "Correct": bool(tp),
        "True Positive": tp,
        "False Positive": fp,
        "False Negative": fn,
        "Residual": residual
    })


    prediction_cache[name] = (padded_waveform, probs_p, probs_noise)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

# --- Stats ---
tp_total = results_df["True Positive"].sum()
fp_total = results_df["False Positive"].sum()
fn_total = results_df["False Negative"].sum()
total_with_picks = sum(
    1 for raw in df["marked_point"]
    if (isinstance(raw, str) and raw.strip().startswith('['))
    or (pd.to_numeric(raw, errors='coerce') is not None and float(raw) >= 0)
)
total_no_picks = df.shape[0] - total_with_picks

print(f"\n✅ Saved results to {OUTPUT_CSV}")
print(f"📌 Total rows with picks: {total_with_picks}")
print(f"📌 Total rows without picks: {total_no_picks}")
print(f"📊 True Positives: {tp_total}")
print(f"📊 False Positives: {fp_total}")
print(f"📊 False Negatives: {fn_total}")
print(f"📊 Waveforms with no predicted picks: {no_predicted_picks}")
print(f"📊 Picks with residual > 1000: {residuals_over_1000}")
print(f"📊 Picks with residual ≤ 1000: {len(residuals_under_1000)}")
if residuals_under_1000:
    print(f"📊 Avg residual (≤1000): {np.mean(residuals_under_1000):.2f}")

# --- T007-specific analysis ---
t007_df = results_df[results_df["Name"].str.contains("Exp_T007")]
num_t007_waveforms = len(t007_df)
num_t007_correct = t007_df["Correct"].sum()

# flatten all residual lists for T007
t007_residuals = []
for r in t007_df["Residual"]:
    if isinstance(r, list):
        t007_residuals.extend(r)
    elif r is not None:
        t007_residuals.append(r)

print("\n🔍 T007 Experiment Metrics")
print(f"📁 Number of T007 waveforms: {num_t007_waveforms}")
print(f"✅ Correctly predicted picks in T007: {num_t007_correct}")
if t007_residuals:
    print(f"📏 Average residual in T007: {np.mean(t007_residuals):.2f} samples")
else:
    print("📏 No residuals available for T007")

# --- Non-T007 residual ---
non_t007_df = results_df[~results_df["Name"].str.contains("Exp_T007")]

non_t007_residuals = []
for r in non_t007_df["Residual"]:
    if isinstance(r, list):
        non_t007_residuals.extend(r)
    elif r is not None:
        non_t007_residuals.append(r)

if non_t007_residuals:
    print(f"\n📏 Average residual (excluding T007): {np.mean(non_t007_residuals):.2f} samples")
else:
    print("\n📏 No residuals available for non-T007 experiments")


# --- Residual histograms (filter out infinities) ---
name_prefix = CHECKPOINT_PATH.split('.')[0]

# only keep finite residuals for the “all” histogram
finite_all = [r for r in all_recorded_residuals if np.isfinite(r)]
if finite_all:
    plt.figure(figsize=(10, 5))
    plt.hist(finite_all, bins=50, edgecolor='black')
    plt.axvline(x=1000, color='red', linestyle='--', label='1000-sample threshold')
    plt.title("Histogram of All Residuals")
    plt.xlabel("Residual (samples)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{name_prefix}_ALL.png", dpi=300)

# residuals_under_1000 never contains inf, so no change there
if residuals_under_1000:
    plt.figure(figsize=(10, 5))
    plt.hist(residuals_under_1000, bins=50, edgecolor='black')
    plt.title("Histogram of Residuals ≤ 1000")
    plt.xlabel("Residual (samples)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{name_prefix}_lt_1000.png", dpi=300)


# --- GUI Viewer ---
class Visualizer:
    def __init__(self, master, results_df, cache):
        self.master = master
        self.master.title("Validation Viewer")
        self.df = results_df
        self.cache = cache
        self.index = 0
        self.total = len(self.df)

        self.frame = tk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(master)
        btn_frame.pack()
        tk.Button(btn_frame, text="Previous", command=self.prev).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Next", command=self.next).pack(side=tk.LEFT)

        self.plot()

    def plot(self):
        self.ax1.clear()
        self.ax2.clear()

        row = self.df.iloc[self.index]
        name = row["Name"]
        true_picks = row["True Pick"]
        try:
            waveform, probs_p, probs_noise = self.cache[name]
            self.ax1.plot(waveform, color="black", label="Waveform")
            # Draw one or multiple true picks
            if isinstance(true_picks, list):
                for i, gt in enumerate(true_picks):
                    self.ax1.axvline(x=gt, color="green", label="True Pick" if i == 0 else "")
            else:
                self.ax1.axvline(x=true_picks, color="green", label="True Pick")

            preds = eval(row["Predicted Picks"]) if isinstance(row["Predicted Picks"], str) else row["Predicted Picks"]
            for p in preds:
                self.ax1.axvline(x=p, color="red", linestyle="--", alpha=0.7)
            self.ax1.set_title(f"{name} ({self.index + 1}/{self.total})")
            self.ax1.legend()

            self.ax2.plot(probs_p, label="P Probability", alpha=0.8)
            self.ax2.plot(probs_noise, label="Noise Probability", alpha=0.8)
            self.ax2.set_title("Model Output Probabilities")
            self.ax2.legend()

            self.canvas.draw()
        except Exception as e:
            self.ax1.set_title(f"⚠️ {e}")
            self.canvas.draw()

    def next(self):
        self.index = (self.index + 1) % self.total
        self.plot()

    def prev(self):
        self.index = (self.index - 1) % self.total
        self.plot()

def on_close():
    print("👋 Exiting...")
    root.destroy()

root = tk.Tk()
root.geometry("1400x700")
app = Visualizer(root, results_df, prediction_cache)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
