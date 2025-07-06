# Rust-ONNX-Chat

Run small open-source Large Language Models (LLMs) locally with blazing-fast inference using ONNX Runtime – all from safe, modern **Rust**.  
This project lets you chat with compact, chat-tuned models like **TinyLlama-1.1B-Chat**, **Qwen-0.5B-Chat**, and **SmolLM2-135M-Instruct** entirely *offline* once they are exported to ONNX.

---

## ✨ Features

* Pure-Rust command-line chat bot (no Python server required)
* Uses the high-performance `ort` crate (ONNX Runtime bindings)
* Supports **stream or pipe input** & interactive REPL
* Easy model switching: `cargo run -- tinyllama | qwen | smollm2`
* Handy special commands:
  * `clear` / `reset` – wipe conversation history
  * send a message containing `model` or `onnx` – check if the model is loaded
* Completely local inference – your data never leaves your machine

---

## 📂 Repository layout

```
├── models/         # Place exported ONNX models here (see below)
├── src/
│   ├── chat.rs     # ChatBot implementation  
│   └── main.rs     # CLI entry-point
└── Cargo.toml
```

---

## 🔧 Prerequisites

| Requirement | Recommended version |
|-------------|---------------------|
| Rust & Cargo | Stable (e.g. 1.78+) |
| Python       | 3.9+ |
| `uv` package manager | latest |

> *Why Python?* Python is used only for one-time to export model using the **Optimum** CLI; the Rust binary itself has **no Python dependency**.

---

## 🚀 Quick start

```bash
# 1) Clone & build
$ git clone https://github.com/ndemir/rust-onnx-chat.git
$ cd rust-onnx-chat
$ cargo build --release    # first build can take a few minutes

# 2) Export a model to ONNX (example: TinyLlama)
$ uv venv                            # create isolated env
$ uv pip install "optimum[onnxruntime]"
$ uv run optimum-cli export onnx \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --task text-generation ./models/tinyllama

# 3) Chat!
$ ./target/release/rust-onnx-chat tinyllama
🤖 ONNX Chat Bot - Type 'quit' to exit
> hello, what is your name?
🤖: hi there! my name is chatbot.
```

---

## 🛠️ Model export details

All supported models follow the same pattern:

```bash
uv run optimum-cli export onnx --model <HF_MODEL_ID> --task text-generation ./models/<local_name>
```

Supported presets out of the box:

| CLI name  | HuggingFace model ID                      |
|-----------|-------------------------------------------|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| `qwen`      | Qwen/Qwen1.5-0.5B-Chat                |
| `smollm2`   | HuggingFaceTB/SmolLM2-135M-Instruct   |

After export, each model directory must contain at minimum:

```
models/<name>/
├── model.onnx                # exported weights
├── tokenizer.json            # HuggingFace tokenizer
├── config.json               # model config
├── tokenizer_config.json     # tokenizer config
├── special_tokens_map.json   # special tokens
└── chat_template.jinja       # (optional) Jinja chat template
```

---

## 🖥️ Usage

Interactive REPL (default model *tinyllama*):

```bash
cargo run --                       # equivalent to cargo run -- tinyllama
```

Choose a specific model:

```bash
cargo run -- qwen
```

Pipe a single prompt:

```bash
echo "Why is the sky blue?" | cargo run -- smollm2
```

### Special commands inside the REPL

| Command                          | Action                                 |
|----------------------------------|----------------------------------------|
| `clear` / `reset`                | Reset conversation context             |
| message containing `model`/`onnx`| Report if the ONNX model is loaded     |
| `quit`                           | Exit the program                       |

---

## 🩺 Troubleshooting

* **OpSet / compatibility warnings** during export are usually safe to ignore.
* If a model fails to load, ensure `model.onnx` exists and matches the architecture.
* On Apple Silicon, you may need to install the `onnxruntime` wheels with `--no-use-system-protobuf` when exporting.
* For build issues, make sure your Rust toolchain is up to date and you have system-level build tools (e.g. `build-essential` on Debian/Ubuntu).

---

## 📜 License

This project is released under the **MIT License**.  
Model weights and licenses remain under their respective upstream authors.

---

Made with ❤️ in Rust. 
