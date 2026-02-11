Markdown
# 🤖 GW AI (Generative Writer)

**A 10M Parameter Language Model built from scratch with PyTorch & Rust.**

GW is a decoder-only Transformer model designed to generate text. It features a custom high-performance tokenizer written in **Rust** (integrated via PyO3) and a **PyTorch** model trained on a mixed dataset of English, Hindi, and Python code.

This project demonstrates the end-to-end process of building an LLM: from raw data processing and custom tokenizer creation to model training and web deployment.

---

## 🚀 Features

- **Hybrid Architecture:** Python for Deep Learning, Rust for high-speed Tokenization.
- **Custom Tokenizer:** Trained on 100MB+ of data using the Unigram algorithm (BPE-style).
- **Transformer Model:** 6 Layers, 6 Heads, 384 Embedding Dimension (approx. 10M parameters).
- **Interactive UI:** A clean web interface built with **Streamlit** for real-time chatting.
- **Configurable:** Adjustable "Temperature" and "Max Length" controls to tune creativity.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Rust (Cargo) installed

### 1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/gw_project.git](https://github.com/YOUR_USERNAME/gw_project.git)
cd gw_project
2. Set Up Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

Bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Compile the Rust Tokenizer
This project uses a custom Rust extension. You must compile it locally.

Bash
cd rust_tok
maturin develop --release
cd ..
🧠 Usage
Run the Web Interface
To chat with GW AI in your browser:

Bash
streamlit run app.py
Train the Model (Optional)
If you want to re-train the model from scratch:

Ensure train.bin and val.bin are in the root directory.

Run the training script:

Bash
python train.py
📂 Project Structure
app.py: The Streamlit web application (Frontend).

model.py: The GPT model architecture (PyTorch nn.Module).

config.py: Hyperparameters and configuration settings.

train.py: The training loop and validation logic.

rust_tok/: The source code for the custom Rust tokenizer.

📜 License
MIT License


### 📝 Step 3: Push the Changes to GitHub
Now that you've created the file, you need to send it to the cloud.

Run these 3 commands in your terminal:

1.  **Add the file:**
    bash
    git add README.md
    
2.  **Commit:**
    bash
    git commit -m "Added professional README"
    ```
3.  **Push:**
    bash
    git push
    ```

**Go refresh your GitHub page.** You should now see this beautiful documentation on the fron
