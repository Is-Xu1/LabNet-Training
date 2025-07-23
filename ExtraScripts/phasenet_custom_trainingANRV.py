import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet
from scipy.signal import find_peaks

# --- Constants ---
SAMPLING_RATE = 5000000
PHASE_LABELS = {"noise": 0, "p": 1}
torch.manual_seed(42)
np.random.seed(42)

# --- Helper Functions ---
def apply_gaussian_mask(length, index, std):
    x = np.arange(length)
    return np.exp(-0.5 * ((x - index) / std) ** 2)

# --- Dataset Class ---
class SlidingWindowDataset(Dataset):
    def __init__(self, root_dir, label_csv, window_size=50000, stride=20000, gauss_std=200,
                 amp_scale_range=(0.8, 1.2), augmented_pick_windows=2):
        self.window_size = window_size
        self.stride = stride
        self.gauss_std = gauss_std
        self.amp_scale_range = amp_scale_range
        self.augmented_pick_windows = augmented_pick_windows
        self.data = []
        self.waveform_cache = {}

        print(f"🔍 Loading picks from: {label_csv}")
        label_df = pd.read_csv(label_csv)
        print(f"📑 Loaded {len(label_df)} pick entries from {label_csv}")
        self.label_dict = label_df.set_index("Name")["marked_point"].to_dict()

        self._cache_and_window_traces(root_dir)
        print(f"✅ Cached {len(self.waveform_cache)} traces into memory.")

        if len(self.data) == 0:
            raise ValueError("❌ No valid windows generated.")

        pick_windows = sum(pick_idx >= 0 for _, _, pick_idx in self.data)
        noise_windows = len(self.data) - pick_windows
        print(f"🪟 Generated {len(self.data)} sliding windows.")
        print(f"🔵 Pick windows: {pick_windows} ({pick_windows / len(self.data) * 100:.2f}%)")
        print(f"⚪ Noise windows: {noise_windows} ({noise_windows / len(self.data) * 100:.2f}%)")
        with open('log.txt', 'a') as file:
            file.write('\n' + str(pick_windows / len(self.data) * 100))

    def _cache_and_window_traces(self, root_dir):
        mseed_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(root_dir)
            for f in files if f.endswith(".mseed")
        ]
        discarded = 0
        for path in mseed_paths:
            try:
                stream = read(path)
                for i, tr in enumerate(stream):
                    name = self._build_name(path, i)
                    waveform = tr.data.astype(np.float32)

                    # Normalize
                    std = waveform.std()
                    if std > 1e-6:
                        waveform = (waveform - waveform.mean()) / std
                    else:
                        waveform = np.zeros_like(waveform)

                    # Pad if shorter than window
                    if len(waveform) < self.window_size:
                        pad_len = self.window_size - len(waveform)
                        waveform = np.concatenate([waveform, np.zeros(pad_len, dtype=np.float32)])

                    self.waveform_cache[name] = waveform
                    pick_idx = self.label_dict.get(name, -1)
                    discarded += self._generate_windows(name, waveform, pick_idx)
            except Exception as e:
                print(f"⚠️ Skipped {path}: {e}")
        print(f"❌ Discarded {discarded} windows with edge picks.")

    def _build_name(self, path, trace_index):
        parts = path.split(os.sep)
        exp = next(p for p in parts if p.startswith("Exp_")).replace("Exp_", "")
        run = next((p for p in parts if p.startswith("Run")), "RunX")
        event = os.path.basename(path).split("_WindowSize")[0]
        return f"p_picks_Exp_{exp}_{run}_{event}_trace{trace_index + 1}"

    def _generate_windows(self, name, waveform, pick_idx):
        L = len(waveform)
        discarded = 0
        for start in range(0, L - self.window_size + 1, self.stride):
            end = start + self.window_size
            if pick_idx >= 0:
                if start + 2000 <= pick_idx < end - 2000:
                    self.data.append((name, start, pick_idx))
                    for _ in range(self.augmented_pick_windows):
                        shift_offset = np.random.randint(-self.window_size // 2, self.window_size // 2)
                        new_start = pick_idx - (self.window_size // 2) + shift_offset
                        new_start = max(0, min(new_start, L - self.window_size))
                        self.data.append((name, new_start, pick_idx))
                elif pick_idx < start - 2000 or pick_idx >= end + 2000:
                    self.data.append((name, start, -1))
                else:
                    discarded += 1
            else:
                self.data.append((name, start, -1))
        return discarded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, start, pick_idx = self.data[idx]
        waveform = self.waveform_cache[name]
        segment = waveform[start:start + self.window_size]

        if len(segment) < self.window_size:
            pad_len = self.window_size - len(segment)
            segment = np.concatenate([segment, np.zeros(pad_len, dtype=np.float32)])

        scale = np.random.uniform(*self.amp_scale_range)
        window = segment * scale

        label = np.zeros((2, self.window_size), dtype=np.float32)
        if pick_idx >= 0:
            local_idx = pick_idx - start
            label[1] = apply_gaussian_mask(self.window_size, local_idx, self.gauss_std)
            label[0] = np.clip(1 - label[1], 0, 1)
        else:
            label[0] = np.ones(self.window_size, dtype=np.float32)

        return torch.tensor(window).unsqueeze(0), torch.tensor(label)

# --- Loss Function ---
class PhaseNetLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, weights=None):
        super().__init__()
        self.eps = eps
        self.weights = weights or torch.tensor([0.1, 1.0])

    def forward(self, prediction, target):
        weights = self.weights.to(prediction.device)
        loss = 0.0
        for c in range(prediction.shape[1]):
            pred = prediction[:, c].clamp(self.eps, 1 - self.eps)
            tgt = target[:, c]
            loss += weights[c] * torch.nn.functional.binary_cross_entropy(pred, tgt, reduction='mean')
        return loss

# --- Validation: Compute TP count ---
def validate_true_positives(model, val_csv, cache, device, tolerance=100):
    val_df = pd.read_csv(val_csv)
    model.eval()
    tp_count = 0
    with torch.no_grad():
        for _, row in val_df.iterrows():
            name = row["Name"]
            if name not in cache:
                continue
            waveform = cache[name]
            x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            y_hat = model(x).squeeze().cpu().numpy()[1]  # P-phase channel
            pred_idx = np.argmax(y_hat)
            true_idx = int(row["marked_point"])
            if abs(pred_idx - true_idx) <= tolerance:
                tp_count += 1
    return tp_count

# --- Training Loop with TP-based Early Stopping ---
def train_model(data_dir, train_csv, val_csv, checkpoint_path, log_csv,
                window_size, gauss_std, epochs=5, batch_size=20, augmented_pick_windows=2,
                patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    dataset = SlidingWindowDataset(data_dir, train_csv, window_size, 20000, gauss_std, (0.9, 1.1), augmented_pick_windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VariableLengthPhaseNet(in_channels=1, classes=2, phases="NP", sampling_rate=SAMPLING_RATE, norm="std").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PhaseNetLoss().to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")

    log = []
    best_tp = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[{checkpoint_path}] Epoch {epoch} Step {i+1}/{len(loader)} → Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            tp_count = validate_true_positives(model, val_csv, dataset.waveform_cache, device)
            print(f"🔍 Validation TP count: {tp_count}")
            if tp_count > best_tp:
                best_tp = tp_count
                no_improve = 0
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
                print(f"💾 Saved checkpoint with TP={tp_count}")
            else:
                no_improve += 1
                print(f"⚠️ No TP improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    print("⏹ Early stopping (TP-based).")
                    break

        log.append({"Epoch": epoch, "Avg Loss": avg_loss})
        pd.DataFrame(log).to_csv(log_csv, index=False)

# --- Main ---
if __name__ == "__main__":
    WINDOWSIZE = [50000]
    GAUSS_STD = [100]
    AUGMENTED_PICKS = [2]
    BATCH_SIZE = [200]

    configs = []
    for w, g, a, b in zip(WINDOWSIZE, GAUSS_STD, AUGMENTED_PICKS, BATCH_SIZE):
        config = {
            "checkpoint": f"{w}w{g}gANRV_{a}.pt",
            "log": f"log_{w}w{g}gANRV_{a}.csv",
            "labels": "p_picks_training.csv",
            "window_size": w,
            "gauss_std": g,
            "augmented_pick_windows": a,
            "batch_size": b
        }
        configs.append(config)

    for config in configs:
        print(f"\n🚀 Starting training for {config['checkpoint']}...\n")
        train_model(
            data_dir=r"f:\Data",
            train_csv=config["labels"],
            val_csv= 'p_picks_validation.csv',
            checkpoint_path=config["checkpoint"],
            log_csv=config["log"],
            window_size=config["window_size"],
            gauss_std=config["gauss_std"],
            epochs=40,
            batch_size=config["batch_size"],
            augmented_pick_windows=config["augmented_pick_windows"]
        )
