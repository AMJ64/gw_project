import streamlit as st
import torch
from torch.nn import functional as F
import rust_tok
import json
import os
import config            # <--- Import Settings
from model import GPT    # <--- Import Brain

# --- CONFIG ---
st.set_page_config(page_title="GW AI", page_icon="🤖")
st.title("🤖 GW: Built from Scratch")
st.caption("A 10M Parameter Language Model built with PyTorch & Rust")

# --- CACHED LOADER ---
@st.cache_resource
def load_resources():
    print("⏳ Loading GW Resources...")
    
    # 1. Load Vocab
    with open(config.TOKENIZER_FILE, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)["vocab"]
    
    sorted_tokens = sorted(vocab_dict.keys())
    token_to_id = {t: i for i, t in enumerate(sorted_tokens)}
    id_to_token = {i: t for i, t in enumerate(sorted_tokens)}
    vocab_size = len(token_to_id)
    
    # 2. Load Rust Tokenizer
    rust_tokenizer = rust_tok.UnigramTok.load(config.TOKENIZER_FILE)

    # 3. Load Model
    model = GPT(vocab_size).to(config.DEVICE)
    if os.path.exists(config.MODEL_FILE):
        state_dict = torch.load(config.MODEL_FILE, map_location=config.DEVICE)
        # Fix keys
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        st.error(f"❌ {config.MODEL_FILE} not found!")
    
    return rust_tokenizer, model, token_to_id, id_to_token

tok, model, token_to_id, id_to_token = load_resources()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
    max_tokens = st.slider("Max Length", 10, 200, 50)

# --- CHAT UI ---
if "messages" not in st.session_state: st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Talk to GW..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Inference Logic
        str_tokens = tok.encode(prompt)
        ids = [token_to_id.get(t, 0) for t in str_tokens]
        ctx = torch.tensor(ids, dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        
        for _ in range(max_tokens):
            idx_cond = ctx[:, -config.BLOCK_SIZE:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            next_id = idx_next.item()
            decoded_token = id_to_token.get(next_id, "").replace(' ', ' ')
            
            full_response += decoded_token
            message_placeholder.markdown(full_response + "▌")
            ctx = torch.cat((ctx, idx_next), dim=1)
            
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})