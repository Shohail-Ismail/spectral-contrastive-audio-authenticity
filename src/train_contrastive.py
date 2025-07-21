import os, random, json, torch, librosa
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# for testing -- [REMOVE LATER]
torch.manual_seed(99)
random.seed(1)

# Load ASV Spoof labels
protocol = {}
with open("data/raw/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt") as pf:
    for line in pf:
        parts = line.strip().split()
        protocol[parts[1]] = 0 if parts[-1] == "bonafide" else 1


# Contrastive dataset with one +ve and -ve pair per reference anchor
class ContrastiveDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.labels = {f: protocol.get(f[:-4], 0) for f in self.files}
        
    # Returns no. of anchor candidates
    def __len__(self): 
        return len(self.files)
    
    # Pick a positive and negative example at random
    def __getitem__(self, idx):
        anchor = self.files[idx]
        la = self.labels[anchor]
        
        pos = random.choice([f for f in self.files if self.labels[f] == la and f != anchor])
        neg = random.choice([f for f in self.files if self.labels[f] != la])
        
        # 1 for positive, 0 for negative
        return (
            os.path.join(self.data_dir, anchor),
            os.path.join(self.data_dir, pos), 1,
            os.path.join(self.data_dir, neg), 0
        )


# Projection model using frozen wav2vec embeddings and 128-dim head
class ProjectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.proj = nn.Linear(self.wav2vec.config.hidden_size, 128)
        
        # Freeze all wav2vec parameters so we only train the small head as 
        # back‑prop through 300 mil parameters is not feasible on CPU
        # and focuses learning on the head
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        
    # Single linear layer mapping 768 to 128 dims
    def forward(self, x):
        hidden = self.wav2vec(x).last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.proj(pooled)

### Contrastive loss function ###
def contrastive_loss(a, b, label, margin=1.0):
    dist = (a - b).pow(2).sum(dim=1).sqrt()
    
    # If label = 1, loss=distance(a, b) => pulled together
    positive = label * dist
    
    # If label = 0, loss=max(0, margin - distance(a, b)) => 
    #pushed apart by at least margin amount
    negative = (1 - label) * torch.clamp(margin - dist, min=0.0)
    return (positive + negative).mean()

# Setup training
device = "cpu"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = ProjectionNet().to(device)
opt = optim.Adam(model.parameters(), lr=1e-4)

dataset = ContrastiveDataset("data/preprocessed/asvspoof")
loader = DataLoader(dataset, bbatch_size=4, shuffle=True, num_workers=4)

# # Train for a tiny subset (limit to 50 anchors)
# LIMIT = 50

print("Starting contrastive training")
losses = []
EPOCHS = 5

for epoch in range(EPOCHS):
    epoch_losses = []
    for idx, (path_a, path_p, _, path_n, _) in enumerate(loader, start=1):
        
        # Unpack single-element batch
        path_a = path_a[0]
        path_p = path_p[0]
        path_n = path_n[0]
        
        # Load audio at 16khz
        y_a, _ = librosa.load(path_a, sr=16000)
        y_p, _ = librosa.load(path_p, sr=16000)
        y_n, _ = librosa.load(path_n, sr=16000)
        
        # Tokenize
        inp_a = processor(y_a, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
        inp_p = processor(y_p, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
        inp_n = processor(y_n, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
        
        # Forward pass
        emb_a = model(inp_a)
        emb_p = model(inp_p)
        emb_n = model(inp_n)
        
        # Positive and negative labels for contrastive loss
        lbl_p = torch.ones_like(emb_a[:,0])
        lbl_n = torch.zeros_like(emb_a[:,0])
        
        # Combine losses
        loss = contrastive_loss(emb_a, emb_p, lbl_p) + contrastive_loss(emb_a, emb_n, lbl_n)
        
        # Backprop (over 128 dims)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        epoch_losses.append(loss.item())
        
    avg_loss = sum(losses)/len(losses)
    print(f"Epoch {epoch + 1} --- avg loss: {avg_loss:.4f}")
    
    

# Save weights and metrics
torch.save(model.state_dict(), "models/contrastive_asvspoof.pth")
with open("results/contrastive_metrics.json", "w") as f:
    if len(losses) > 0:
        avg_loss = sum(losses)/len(losses)
    else:
        avg_loss = None
    json.dump({"avg_loss": avg_loss}, f)
print("ouch my brain 29. also model and metrics saved.")
