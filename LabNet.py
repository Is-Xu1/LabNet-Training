import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet
from scipy.signal import find_peaks

SAMPLING_RATE = 5_000_000

def apply_gaussian_mask(length, index, std=100):
    """
    Apply a Gaussian mask centered at a specific index within a given length.

    Args:
        length (int): Total length of the output array.
        index (int): The center index for the Gaussian mask.
        std (int, optional): Standard deviation of the Gaussian distribution. Default is 100.

    Returns:
        np.ndarray: A 1D NumPy array of shape (length,) with Gaussian weights applied.
    """
    x = np.arange(length)
    return np.exp(-0.5 * ((x - index) / std) ** 2)

class PhaseNetLoss(torch.nn.Module):
    """
    
    """
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

class SlidingWindowDataset(Dataset):
    def __init__(self, root_dir, label_csv, window_size=50000, stride=20000, gauss_std=200,
                 amp_scale_range=(0.8, 1.2), filter_config=None, augmented_pick_windows=2):
        self.window_size = window_size
        self.stride = stride
        self.gauss_std = gauss_std
        self.amp_scale_range = amp_scale_range
        self.filter_config = filter_config or {"lowcut": 1e5, "highcut": 1.5e6, "order": 4}
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
            raise ValueError("❌ No valid windows generated. Check data folder and labels.")

        pick_windows = sum(pick_idx >= 0 for _, _, pick_idx in self.data)
        noise_windows = len(self.data) - pick_windows

        print(f"🪟 Generated {len(self.data)} sliding windows.")
        print(f"🔵 Pick windows: {pick_windows} ({pick_windows / len(self.data) * 100:.2f}%)")
        print(f"⚪ Noise windows: {noise_windows} ({noise_windows / len(self.data) * 100:.2f}%)")
        with open('log.txt', 'a') as file:
            file.write('\n'+str(pick_windows / len(self.data) * 100))

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

                    # Step 1: Apply bandpass filter (removed)
            

                    # Step 2: Global normalization
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
                    # Original pick window
                    self.data.append((name, start, pick_idx))

                    # 🔍 Random shift augmentation for picks
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

        # Step 3: Amplitude scaling (augmentation)
        scale = np.random.uniform(*self.amp_scale_range)
        window = segment * scale

        # Label creation
        label = np.zeros((2, self.window_size), dtype=np.float32)
        if pick_idx >= 0:
            local_idx = pick_idx - start
            label[1] = apply_gaussian_mask(self.window_size, local_idx, self.gauss_std)
            label[0] = np.clip(1 - label[1], 0, 1)
        else:
            label[0] = np.ones(self.window_size, dtype=np.float32)

        return torch.tensor(window, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32)
    

def train_model(data_dir, label_csv, checkpoint_path, log_csv, window_size, gauss_std, SAMPLING_RATE,
                epochs=5, batch_size=20, filter_config=None, augmented_pick_windows=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    dataset = SlidingWindowDataset(
        root_dir=data_dir,
        label_csv=label_csv,
        window_size=window_size,
        stride=20000,
        gauss_std=gauss_std,
        amp_scale_range=(0.9, 1.1),
        filter_config=filter_config,
        augmented_pick_windows=augmented_pick_windows
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VariableLengthPhaseNet(
        in_channels=1,
        classes=2,
        phases="NP",
        sampling_rate=SAMPLING_RATE,
        norm="std"
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PhaseNetLoss().to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print("ℹ️ No checkpoint found, training from scratch.")

    if os.path.exists(log_csv):
        log_df = pd.read_csv(log_csv)
        log = log_df.to_dict("records")
        start_epoch = log_df["Epoch"].max() + 1
    else:
        log = []
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[{checkpoint_path}] Epoch {epoch} Step {i+1}/{len(loader)} → Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ [{checkpoint_path}] Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, checkpoint_path)
        print(f"💾 Saved checkpoint to {checkpoint_path}")

        log.append({"Epoch": epoch, "Avg Loss": avg_loss})
        pd.DataFrame(log).to_csv(log_csv, index=False)

def parse_name(name: str):
    """
    Parse a waveform name into experiment, run, event, and trace index.

    Args:
        name (str): Example format: "p_picks_Exp_T007_Run1_Event_4_trace1"

    Returns:
        tuple: (experiment, run, event, trace_index)
    """
    parts = name.split("_")
    trace_index = int(parts[-1].replace("trace", "")) - 1
    event = f"{parts[-3]}_{parts[-2]}"
    exp = f"{parts[2]}_{parts[3]}"
    run = "_".join(parts[4:-3])
    return exp, run, event, trace_index


def load_waveform(name_key: str, data_dir: str):
    """
    Load a waveform by its name key from the dataset directory.

    Args:
        name_key (str): Name from the labels CSV.
        data_dir (str): Base directory of the waveform data.

    Returns:
        tuple: (waveform: np.ndarray, full_path: str)
    """
    exp, run, event, trace_index = parse_name(name_key)
    base_folder = os.path.join(data_dir, exp)
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


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """
    Normalize waveform using standard deviation normalization.
    Falls back to zeros if std is too small.

    Args:
        waveform (np.ndarray): Input waveform.

    Returns:
        np.ndarray: Normalized waveform.
    """
    std = waveform.std()
    if std > 1e-6:
        return (waveform - waveform.mean()) / (std + 1e-6)
    return np.zeros_like(waveform)


def sliding_window_inference(model, waveform, window_size=50_000, stride=20_000, device="cpu"):
    """
    Perform sliding window inference on a waveform.

    Args:
        model (torch.nn.Module): Trained model.
        waveform (np.ndarray): Input waveform.
        window_size (int): Size of the inference window.
        stride (int): Step size for sliding window.
        device (str): Torch device.

    Returns:
        tuple: (probs_p, probs_noise)
    """
    probs = np.zeros((2, len(waveform)))
    count = np.zeros(len(waveform))

    for start in range(0, len(waveform) - window_size + 1, stride):
        segment = waveform[start:start + window_size]
        input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)[0].detach().cpu().numpy()
        probs[:, start:start + window_size] += output
        count[start:start + window_size] += 1

    last_start = len(waveform) - window_size
    if last_start % stride != 0:
        remaining = waveform[last_start:]
        padded = np.zeros(window_size, dtype=np.float32)
        padded[:len(remaining)] = remaining
        input_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)[0].detach().cpu().numpy()
        probs[:, last_start:] += output[:, :len(remaining)]
        count[last_start:] += 1

    count[count == 0] = 1
    probs /= count
    return probs[1], probs[0]  # P-probability, Noise-probability


def detect_p_picks(probs_p, threshold=0.5, min_distance=3000):
    """
    Detect P-wave picks using peak detection on probability trace.

    Args:
        probs_p (np.ndarray): Probability trace for P-phase.
        threshold (float): Peak detection threshold.
        min_distance (int): Minimum distance between peaks.

    Returns:
        np.ndarray: Indices of detected peaks.
    """
    peaks, _ = find_peaks(probs_p, height=threshold, prominence=0.3, distance=min_distance)
    return peaks


def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load a trained VariableLengthPhaseNet model from checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to load the model onto.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    model = VariableLengthPhaseNet(
        in_channels=1,
        classes=2,
        phases="NP",
        sampling_rate=SAMPLING_RATE,
        norm="std"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model