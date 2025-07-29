import LabNet as lb
import torch
from torch.utils.data import DataLoader

# Training

# Create dataset
dataset = lb.SlidingWindowDataset(
    root_dir="f:/Data",                  # Path to waveform data
    label_csv="p_picks_Data.csv",       # CSV containing picks
    window_size=50000,
    stride=20000,
    gauss_std=100,
    augmented_pick_windows=2
)

# DataLoader
loader = DataLoader(dataset, batch_size=20, shuffle=True)

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = lb.load_model("50000w100gANR_2.pt", device=device)  # Load existing checkpoint or new model

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = lb.PhaseNetLoss().to(device)

# Training loop (example: 2 epochs)
for epoch in range(2):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Avg Loss = {total_loss / len(loader):.4f}")

# Save trained model
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "trained_model.pt")
print("✅ Model saved as trained_model.pt")


# Inference

# Load trained model for inference
model = lb.load_model("trained_model.pt", device=device)

# Load and preprocess a single waveform
waveform, _ = lb.load_waveform("p_picks_Exp_T007_Run1_Event_4_trace1", "f:/Data")
waveform = lb.normalize_waveform(waveform)

# Run sliding window inference
probs_p, probs_noise = lb.sliding_window_inference(
    model, waveform, window_size=50000, stride=20000, device=device
)

# Detect P picks
picks = lb.detect_p_picks(probs_p, threshold=0.5, min_distance=3000)
print("Predicted picks:", picks)
