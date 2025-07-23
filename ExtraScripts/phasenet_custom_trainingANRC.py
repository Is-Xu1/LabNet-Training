import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from obspy import read
from seisbench.models import VariableLengthPhaseNet

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
    def __init__(self, root_dir, label_csv, window_size=50000, stride=20000,
                 gauss_std=200, amp_scale_range=(0.8, 1.2), augmented_pick_windows=2):
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

        pick_windows = sum(pick_idx >= 0 for _, _, pick_idx in self.data)
        noise_windows = len(self.data) - pick_windows
        print(f"🪟 Generated {len(self.data)} sliding windows.")
        print(f"🔵 Pick windows: {pick_windows} ({pick_windows / len(self.data) * 100:.2f}%)")
        print(f"⚪ Noise windows: {noise_windows} ({noise_windows / len(self.data) * 100:.2f}%)")

    def _cache_and_window_traces(self, root_dir):
        mseed_paths = [os.path.join(root, f)
                       for root, _, files in os.walk(root_dir)
                       for f in files if f.endswith('.mseed')]
        discarded = 0
        for path in mseed_paths:
            try:
                stream = read(path)
                for i, tr in enumerate(stream):
                    name = self._build_name(path, i)
                    waveform = tr.data.astype(np.float32)

                    # Global normalization
                    std = waveform.std()
                    if std > 1e-6:
                        waveform = (waveform - waveform.mean()) / std
                    else:
                        waveform = np.zeros_like(waveform)

                    # Pad if shorter
                    if len(waveform) < self.window_size:
                        pad = self.window_size - len(waveform)
                        waveform = np.concatenate([waveform, np.zeros(pad, dtype=np.float32)])

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
            if pick_idx >= 0 and start + 2000 <= pick_idx < end - 2000:
                self.data.append((name, start, pick_idx))
                for _ in range(self.augmented_pick_windows):
                    shift = np.random.randint(-self.window_size // 2, self.window_size // 2)
                    new_start = pick_idx - self.window_size // 2 + shift
                    new_start = max(0, min(new_start, L - self.window_size))
                    self.data.append((name, new_start, pick_idx))
            elif pick_idx < start - 2000 or pick_idx >= end + 2000:
                self.data.append((name, start, -1))
            else:
                discarded += 1
        return discarded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, start, pick_idx = self.data[idx]
        seg = self.waveform_cache[name][start:start + self.window_size]
        scale = np.random.uniform(*self.amp_scale_range)
        window = seg * scale
        label = np.zeros((2, self.window_size), dtype=np.float32)
        if pick_idx >= 0:
            local_idx = pick_idx - start
            label[1] = apply_gaussian_mask(self.window_size, local_idx, self.gauss_std)
            label[0] = np.clip(1 - label[1], 0, 1)
        else:
            label[0] = np.ones(self.window_size, dtype=np.float32)
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

# --- Loss Function ---
class PhaseNetLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, weights=None):
        super().__init__()
        self.eps = eps
        self.weights = weights or torch.tensor([0.1, 1.0])
    def forward(self, pred, tgt):
        w = self.weights.to(pred.device)
        loss = 0.0
        for c in range(pred.shape[1]):
            p = pred[:, c].clamp(self.eps, 1 - self.eps)
            loss += w[c] * torch.nn.functional.binary_cross_entropy(p, tgt[:, c], reduction='mean')
        return loss

# --- Training Loop (Curriculum + Resume/Save) ---
def train_model(data_dir, easy_csv, hard_csv, resume_ckpt, save_ckpt, log_csv,
                window_size, gauss_std, epochs=40, batch_size=200,
                augmented_pick_windows=2, curriculum_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    # Easy + hard datasets
    ds_easy = SlidingWindowDataset(data_dir, easy_csv, window_size, stride=20000,
                                   gauss_std=gauss_std, amp_scale_range=(0.8,1.2),
                                   augmented_pick_windows=augmented_pick_windows)
    ds_hard = SlidingWindowDataset(data_dir, hard_csv, window_size, stride=20000,
                                   gauss_std=gauss_std, amp_scale_range=(0.8,1.2),
                                   augmented_pick_windows=augmented_pick_windows)
    ds_all = ConcatDataset([ds_easy, ds_hard])

    model = VariableLengthPhaseNet(in_channels=1, classes=2, phases="NP",
                                   sampling_rate=SAMPLING_RATE, norm="std").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PhaseNetLoss().to(device)

    # Resume from checkpoint
    start_epoch = 0
    log = []
    if os.path.exists(resume_ckpt):
        chk = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(chk["model"])
        optimizer.load_state_dict(chk["optimizer"])
        print(f"✅ Resumed from {resume_ckpt}")
    if os.path.exists(log_csv):
        df = pd.read_csv(log_csv); log = df.to_dict('records'); start_epoch = df['Epoch'].max()+1

    for epoch in range(start_epoch, start_epoch + epochs):
        # Linear curriculum up to 80%
        base = min((epoch + 1)/curriculum_epochs, 1.0)
        ratio = base * 0.5
        print(f"🎓 Epoch {epoch}: hard ratio={ratio:.3f}")

        w_e = np.ones(len(ds_easy))*(1-ratio)/len(ds_easy)
        w_h = np.ones(len(ds_hard))*ratio/len(ds_hard)
        sampler = WeightedRandomSampler(np.concatenate([w_e, w_h]),
                                        num_samples=len(ds_easy)+len(ds_hard),
                                        replacement=True)
        loader = DataLoader(ds_all, batch_size=batch_size, sampler=sampler)

        total_loss = 0.0
        model.train()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[{save_ckpt}] Epoch {epoch} Step {i+1}/{len(loader)} → Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ [{save_ckpt}] Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")

        # Save to new checkpoint
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, save_ckpt)
        print(f"💾 Saved checkpoint as {save_ckpt}")

        log.append({"Epoch": epoch, "Avg Loss": avg_loss})
        pd.DataFrame(log).to_csv(log_csv, index=False)


# --- Main ---
if __name__ == "__main__":
    WINDOWSIZE = [50000]
    GAUSS_STD = [100]
    AUGMENTED_PICKS = [2]
    BATCH_SIZE = [200]
    EASY_CSV = "50000w100gANR_2_easy.csv"
    HARD_CSV = "50000w100gANR_2_hard.csv"
    RESUME_CKPT = "50000w100gANRC_2.pt"
    LOG_CSV     = "log_50000w100gANRC_2.csv"

    configs = []
    for w, g, a, b in zip(WINDOWSIZE, GAUSS_STD, AUGMENTED_PICKS, BATCH_SIZE):
        config = {
            "checkpoint": f"{w}w{g}gANRC_{a}.pt",
            "log": f"log_{w}w{g}gANRC_{a}.csv",
            "window_size": w,
            "gauss_std": g,
            "augmented_pick_windows": a,  # 🔍 Add more augmented windows per pick
            "batch_size": b
        }
        configs.append(config)

    for config in configs:
        print(f"\n🚀 Starting training for {config['checkpoint']}...\n")
        train_model(
            data_dir=r"f:\Data",
            easy_csv=EASY_CSV,
            hard_csv=HARD_CSV,
            resume_ckpt=RESUME_CKPT,
            save_ckpt=config["checkpoint"],
            log_csv=config["log"],
            window_size=config["window_size"],
            gauss_std=config["gauss_std"],
            epochs=40,
            batch_size=config["batch_size"],
            augmented_pick_windows=config["augmented_pick_windows"],
            curriculum_epochs=1
        )
