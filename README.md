# Introduction to NLP Assignment 3: Cipher Decryption & Language Modeling

This repository implements neural sequence models built from scratch in PyTorch (without using built-in `nn.RNN` or `nn.LSTM` abstractions) for cipher decryption, language modeling, and error-rectification pipelines.

---

## Key Implementations

### Task 1: Cipher Decryption
* **Models**: Custom **RNN** and **LSTM** sequence models trained to map encrypted ciphertext (`cipher_00.txt`) back to plaintext (`plain.txt`).
* **Metrics**: Character Accuracy, Word Accuracy, and Levenshtein Distance.

### Task 2: Language Modeling
* **Models**:
  * **Bi-LSTM**: Bidirectional LSTM for Masked Language Modeling (MLM).
  * **SSM**: Simple State Space Model for Next Word Prediction (NWP).
* **Metrics**: Perplexity.

### Task 3: Error Correction Pipeline
* **Pipeline**: Combines the trained decryption model (LSTM) with language models (Bi-LSTM / SSM) to rectify decryption errors caused by noise (`cipher_00.txt` – `cipher_05.txt`).
* **Metrics**: Accuracy, Levenshtein Distance, Perplexity, BLEU, and ROUGE-1 / ROUGE-2 / ROUGE-L.

---

## Directory Structure

```
.
├── main.py              # Central entrypoint for running tasks
├── config/              # Task and model YAML configurations
├── data/                # Plaintext and ciphertext datasets (cipher_00 to cipher_05)
├── outputs/             # Model evaluation logs, plots, and output text files
└── src/
    ├── common/          # Custom neural cells/layers (models.py), datasets, & metrics
    ├── task1/           # Task 1 training & evaluation scripts (RNN/LSTM)
    ├── task2/           # Task 2 training & evaluation scripts (Bi-LSTM/SSM)
    ├── task3/           # Task 3 error correction pipeline scripts
    └── utils/           # Checkpoints, HuggingFace, & WandB helpers
```

---

## Setup & Execution

### Setup
Install dependencies and sync environment using `uv`:
```bash
uv sync
```

### Running Tasks
All tasks are run using `main.py`:
```bash
uv run main.py <task> [--mode {train,evaluate,both}] [--config CONFIG_PATH]
```

#### Available Tasks:
* `task1_rnn`: Task 1 RNN Decryption Model
* `task1_lstm`: Task 1 LSTM Decryption Model
* `task2_bilstm`: Task 2 Bi-LSTM MLM Model
* `task2_ssm`: Task 2 SSM NWP Model
* `task3_bilstm`: Task 3 Pipeline with Bi-LSTM
* `task3_ssm`: Task 3 Pipeline with SSM

#### Examples:
```bash
# Evaluate Task 1 LSTM model
uv run main.py task1_lstm --mode evaluate

# Train Task 2 SSM model
uv run main.py task2_ssm --mode train

# Run Task 3 Error Correction Pipeline
uv run main.py task3_bilstm --mode evaluate
```

---

## Model Architectures Overview

All sequence models are built from scratch using primitive PyTorch layers (`nn.Linear`, `nn.Embedding`, custom loops):

* **Custom RNN (`CustomRNNCell` / `DecryptionModel`)**:
  * **Recurrence**: $h_t = \tanh(W_{x} x_t + W_{h} h_{t-1})$
  * **Structure**: Embedding $\rightarrow$ Stacked `CustomRNNLayer`s (with residual connections) $\rightarrow$ Dropout $\rightarrow$ Linear output head.
* **Custom LSTM (`CustomLSTMCell` / `DecryptionModel`)**:
  * **Gates & State**: Concatenates $[x_t, h_{t-1}]$ to compute forget ($f_t$), input ($i_t$), candidate ($\tilde{c}_t$), and output ($o_t$) gates. Updates cell state $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ and hidden state $h_t = o_t \odot \tanh(c_t)$.
  * **Structure**: Embedding $\rightarrow$ Stacked `CustomRNNLayer`s (LSTM mode, with residual connections) $\rightarrow$ Dropout $\rightarrow$ Linear output head.
* **Custom Bi-LSTM (`CustomBiLSTM`)**:
  * **Structure**: Embedding $\rightarrow$ Independent forward and backward custom LSTM passes $\rightarrow$ Concatenation $[h_{\text{fw}}, h_{\text{bw}}]$ $\rightarrow$ Dropout $\rightarrow$ Linear head. Used for Masked Language Modeling (MLM).
* **State Space Model (`SimpleSSM`)**:
  * **Recurrence**: $s_t = \tanh(s_{t-1} A + e_t B)$ using learnable transition matrix $A$ and projection matrix $B$.
  * **Structure**: Embedding $\rightarrow$ Recurrent state step $\rightarrow$ Dropout $\rightarrow$ Linear output head. Used for Next Word Prediction (NWP).