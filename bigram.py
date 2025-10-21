# Libraries
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 1000
LEARNING_RATE = 1e-2
EVAL_ITERS = 200
TRAIN_VAL_SPLIT = 0.9
EVAL_ITERS = 300
PRINT_INTERVAL = 50
MAX_NEW_TOKENS = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
dataset = load_dataset("FrancophonIA/french_rap_lyrics_small")
texts = dataset['train'][::]['text']
text = "\n".join(texts)

chars = sorted(list(set(text)))
vocab_size = len(chars) 

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Train & val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(TRAIN_VAL_SPLIT * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram Model
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx) # (B,T,C)

    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):
    if iter % PRINT_INTERVAL == 0:
        losses = estimate_loss()
        print(f'Step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
        
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
# Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()))