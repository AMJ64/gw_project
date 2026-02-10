import torch
import numpy as np
import time
import os
import config            # <--- New: Settings
from model import GPT    # <--- New: Architecture

# --- 1. SETUP ---
print(f"⚙️  Device: {config.DEVICE}")
if config.DEVICE == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

# --- 2. LOAD DATA ---
if not os.path.exists(config.TRAIN_BIN):
    raise FileNotFoundError(f"❌ Missing {config.TRAIN_BIN}. Run data preparation first!")

train_data = np.fromfile(config.TRAIN_BIN, dtype=np.uint16)
val_data = np.fromfile(config.VAL_BIN, dtype=np.uint16)

# Get vocab size dynamically
vocab_size = max(np.max(train_data), np.max(val_data)) + 1
print(f"📚 Vocab Size: {vocab_size}")

# --- 3. HELPER FUNCTIONS ---
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+config.BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.BLOCK_SIZE]).astype(np.int64)) for i in ix])
    x, y = x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.EVAL_INTERVAL)
        for k in range(config.EVAL_INTERVAL):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 4. TRAINING LOOP ---
def train():
    # Initialize Model
    model = GPT(vocab_size)
    model = model.to(config.DEVICE)
    
    # Compilation (Optional, great for T4/A100 GPUs)
    # model = torch.compile(model) 

    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    print(f"🚀 Training started for {config.MAX_ITERS} steps...")
    start_time = time.time()

    for iter in range(config.MAX_ITERS):
        # Evaluation
        if iter % config.EVAL_INTERVAL == 0 or iter == config.MAX_ITERS - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Training Step
        xb, yb = get_batch('train')
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    end_time = time.time()
    print(f"🏁 Finished in {(end_time - start_time)/60:.2f} minutes.")

    # Save
    torch.save(model.state_dict(), config.MODEL_FILE)
    print(f"💾 Model saved to {config.MODEL_FILE}")

if __name__ == "__main__":
    train()