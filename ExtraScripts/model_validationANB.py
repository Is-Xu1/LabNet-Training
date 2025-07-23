import os
import numpy as np
import pandas as pd
import torch
from obspy import read
from seisbench.models import VariableLengthPhaseNet
from scipy.signal import find_peaks, butter, filtfilt
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Config ---
WINDOW_SIZE = 50000
STRIDE = 20000
SAMPLING_RATE = 5000000
THRESHOLD = 0.5
TOLERANCE = 50
CHECKPOINT_PATH = "50000w100gANBR_200l1000k4o.pt"
DATA_DIR = "f:/Data"
VALIDATION_CSV = "p_picks_validation.csv"
OUTPUT_CSV = f"{CHECKPOINT_PATH}.csv"

# Bandpass filter settings (same as training)
FILTER_CONFIG = {"lowcut": 200, "highcut": 1e6, "order": 4}

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

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Apply Butterworth bandpass filter to data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid normalized frequencies: low={low}, high={high}")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

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
    pick_idx = int(row["marked_point"])
    has_pick = pick_idx >= 0

    try:
        waveform, _ = load_waveform(name)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        continue

    original_length = len(waveform)

    # Step 1: Bandpass filter (same as training)
    original_waveform = waveform
    waveform = bandpass_filter(
        waveform,
        fs=SAMPLING_RATE,
        lowcut=FILTER_CONFIG["lowcut"],
        highcut=FILTER_CONFIG["highcut"],
        order=FILTER_CONFIG["order"]
    )

    # --- Zero-pad short waveforms ---
    if original_length < WINDOW_SIZE:
        padded_waveform = np.zeros(WINDOW_SIZE, dtype=np.float32)
        padded_waveform[:original_length] = waveform
        waveform = padded_waveform
    else:
        padded_waveform = waveform.copy()

    # Step 2: Normalize entire waveform once
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
    matched = False
    tp, fp, fn = 0, 0, 0
    residual = None

    # Residual tracking
    if has_pick and len(peak_indices) > 0:
        all_residuals = [abs(p - pick_idx) for p in peak_indices]
        min_residual = min(all_residuals)
        residual = min_residual
        all_recorded_residuals.append(min_residual)
        if min_residual <= 1000:
            residuals_under_1000.append(min_residual)
        else:
            residuals_over_1000 += 1

    # Confusion counts
    if len(peak_indices) == 0:
        no_predicted_picks += 1
        if has_pick:
            fn = 1
        else:
            tp = 1  # correctly predicted no pick
    else:
        for p in peak_indices:
            if has_pick and abs(p - pick_idx) <= TOLERANCE and not matched:
                tp += 1
                matched = True
            elif has_pick and abs(p - pick_idx) <= 1000:
                continue
            elif not has_pick:
                fp += 1
            else:
                fp += 1
        if has_pick and not matched:
            fn = 1

    results.append({
        "Name": name,
        "True Pick": pick_idx,
        "Predicted Picks": peak_indices.tolist(),
        "Correct": bool(tp),
        "True Positive": tp,
        "False Positive": fp,
        "False Negative": fn,
        "Residual": residual
    })

    prediction_cache[name] = (original_waveform, waveform, probs_p, probs_noise)


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

# --- Stats ---
tp_total = results_df["True Positive"].sum()
fp_total = results_df["False Positive"].sum()
fn_total = results_df["False Negative"].sum()
total_with_picks = df[df["marked_point"] >= 0].shape[0]
total_no_picks = df[df["marked_point"] < 0].shape[0]

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
t007_with_residuals = t007_df[t007_df["Residual"].notnull()]
avg_t007_residual = t007_with_residuals["Residual"].mean() if not t007_with_residuals.empty else None

print("\n🔍 T007 Experiment Metrics")
print(f"📁 Number of T007 waveforms: {num_t007_waveforms}")
print(f"✅ Correctly predicted picks in T007: {num_t007_correct}")
if avg_t007_residual is not None:
    print(f"📏 Average residual in T007: {avg_t007_residual:.2f} samples")
else:
    print("📏 No residuals available for T007")

# --- Non-T007 residual ---
non_t007_df = results_df[~results_df["Name"].str.contains("Exp_T007")]
non_t007_with_residuals = non_t007_df[non_t007_df["Residual"].notnull()]
if not non_t007_with_residuals.empty:
    avg_non_t007_residual = non_t007_with_residuals["Residual"].mean()
    print(f"\n📏 Average residual (excluding T007): {avg_non_t007_residual:.2f} samples")
else:
    print("\n📏 No residuals available for non-T007 experiments")

# --- Residual histograms ---
name_prefix = CHECKPOINT_PATH.split('.')[0]
if all_recorded_residuals:
    plt.figure(figsize=(10, 5))
    plt.hist(all_recorded_residuals, bins=50, color='mediumslateblue', edgecolor='black')
    plt.axvline(x=1000, color='red', linestyle='--', label='1000-sample threshold')
    plt.title("Histogram of All Residuals")
    plt.xlabel("Residual (samples)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{name_prefix}_ALL.png", dpi=300)

if residuals_under_1000:
    plt.figure(figsize=(10, 5))
    plt.hist(residuals_under_1000, bins=50, color='seagreen', edgecolor='black')
    plt.title("Histogram of Residuals ≤ 1000")
    plt.xlabel("Residual (samples)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{name_prefix}_lt_1000.png", dpi=300)


# --- GUI Viewer with Linked Zoom/Pan ---
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

        # Create figure with 3 subplots sharing X-axis for synchronized zoom/pan
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        self.fig.subplots_adjust(hspace=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation buttons
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=10)
        prev_btn = tk.Button(btn_frame, text="⬅ Previous", command=self.prev, width=12)
        prev_btn.pack(side=tk.LEFT, padx=10)
        next_btn = tk.Button(btn_frame, text="Next ➡", command=self.next, width=12)
        next_btn.pack(side=tk.LEFT, padx=10)

        self.plot()

    def plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        row = self.df.iloc[self.index]
        name = row["Name"]
        true_pick = row["True Pick"]

        try:
            # Retrieve original waveform, preprocessed waveform, and probabilities
            original_waveform, preprocessed_waveform, probs_p, probs_noise = self.cache[name]

            # 1. Original waveform
            self.ax1.plot(original_waveform, color="black")
            self.ax1.axvline(x=true_pick, color="green", linestyle="-", label="True Pick")
            preds = eval(row["Predicted Picks"]) if isinstance(row["Predicted Picks"], str) else row["Predicted Picks"]
            for p in preds:
                self.ax1.axvline(x=p, color="red", linestyle="--", alpha=0.7)
            self.ax1.set_title(f"Original Waveform ({self.index + 1}/{self.total})")
            self.ax1.legend()

            # 2. Preprocessed waveform
            self.ax2.plot(preprocessed_waveform, color="blue")
            self.ax2.set_title("Preprocessed Waveform (Bandpass + Normalization)")

            # 3. Model probabilities
            self.ax3.plot(probs_p, label="P Probability", color="blue")
            self.ax3.plot(probs_noise, label="Noise Probability", color="gray")
            self.ax3.set_title("Model Output Probabilities")
            self.ax3.legend()

            # Ensure x-axis limits stay linked
            self.fig.canvas.draw_idle()
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


# ✅ Update cache during inference:
# prediction_cache[name] = (padded_waveform, waveform, probs_p, probs_noise)

# --- Start GUI ---
def on_close():
    print("👋 Exiting...")
    root.destroy()

root = tk.Tk()
root.geometry("1400x800")
app = Visualizer(root, results_df, prediction_cache)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
