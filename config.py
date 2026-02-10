import torch

# Device config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
BLOCK_SIZE = 256       # Context length
N_EMBD = 384           # Embedding dimension
N_HEAD = 6             # Attention heads
N_LAYER = 6            # Transformer layers
DROPOUT = 0.2          # Regularization

# Training specific
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500

# File paths
TOKENIZER_FILE = "tokenizer_multi.json"
MODEL_FILE = "model_day10.pt"
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"