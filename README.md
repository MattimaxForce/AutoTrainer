# 🧠 AutoTrainer Wizard  
**Automated AI Model Training — Simplified, Powerful, Adaptable**

AutoTrainer Wizard is an **automatic trainer for artificial intelligence models**.  
It allows you to configure, optimize, and launch model training with just a few clicks — or let it handle everything automatically, adapting to your GPU or CPU.

---

## 🚀 Main Features

- Simple, intuitive graphical interface.  
- Support for **HuggingFace models** (e.g., `bert-base-uncased`, `gpt2`, etc.).  
- Compatibility with **local datasets** or **HuggingFace Datasets Hub** (`username/dataset-name`).  
- Manual or automatic selection of:
  - **Optimizer:** `AdamW`, `SGD`, `Adafactor`
  - **Precision:** `FP32`, `FP16`, `BF16`
  - **Learning rate**, **scheduler**, **batch size**, **epochs**, **gradient accumulation**
- **AutoTrain mode**: for non-experts, the system automatically adjusts everything based on your hardware.
- Supports CPU, GPU, and mixed-precision training.

---

## 🧩 Installation

You can install and use AutoTrainer Wizard in two ways.

### ✅ Method 1 — Direct Download (easiest)

1. Go to the GitHub repository:  
   🔗 [https://github.com/MattimaxForce/AutoTrainer](https://github.com/MattimaxForce/AutoTrainer)
2. Click **Code → Download ZIP**
3. Extract the ZIP file into a local folder
4. Open a terminal in that folder
5. Install the dependencies:
   ```bash
   pip install -r requirements.txt

6. Launch the application:

   * Double-click on `main.py`
     **or**
   * From terminal:

     ```bash
     python main.py
     ```

---

### 💻 Method 2 — Clone via Git

If you prefer to use Git:

```bash
git clone https://github.com/MattimaxForce/AutoTrainer.git
cd AutoTrainer
pip install -r requirements.txt
python main.py
```

---

## 🧠 Graphical Interface (UI Overview)

### HuggingFace

> **AutoTrain Wizard**
> Automate your model training with HuggingFace

#### 🔧 Model Setup

| Field            | Description                            | Example                 |
| ---------------- | -------------------------------------- | ----------------------- |
| **Model Name**   | HuggingFace model name or identifier   | `bert-base-uncased`     |
| **Dataset Path** | Local path or HuggingFace dataset name | `username/dataset-name` |

---

#### ⚙️ Training Configuration

| Parameter         | Available Options              | Default  |
| ----------------- | ------------------------------ | -------- |
| **Optimizer**     | `AdamW`, `SGD`, `Adafactor`    | `AdamW`  |
| **Learning Rate** | Numeric value (e.g., `0.0001`) | `0.0001` |
| **Scheduler**     | `Linear`                       | `Linear` |
| **Precision**     | `FP32`, `FP16`, `BF16`         | `FP32`   |

---

#### 📈 Training Parameters

| Parameter                 | Description                 | Default |
| ------------------------- | --------------------------- | ------- |
| **Epochs**                | Number of training epochs   | `3`     |
| **Batch Size**            | Batch size for training     | `8`     |
| **Gradient Accumulation** | Gradient accumulation steps | `2`     |

---

#### ▶️ Start Training

Click **Start Training** to begin.
The trainer will automatically detect your hardware (GPU/CPU) and adjust precision, batch size, and parameters for optimal performance.

---

## ⚡ Quick Usage Example

```bash
python main.py
```

In the application window:

1. Enter:

   * **Model Name:** `bert-base-uncased`
   * **Dataset Path:** `username/dataset-name`
2. Leave Auto mode enabled or manually adjust parameters.
3. Click **Start Training**.
4. Wait for the process to complete — results will be saved in the `outputs/` directory.

---

## 🧰 Requirements

* Python ≥ 3.8
* Libraries listed in `requirements.txt`
* NVIDIA GPU (optional, recommended)
* Internet connection to download models and datasets from HuggingFace

---

## 📂 Project Structure

```
AutoTrainer/
├── main.py
├── requirements.txt
├── ui/
│   ├── components/
│   └── assets/
├── trainer/
│   ├── train.py
│   ├── utils.py
│   └── config.py
└── outputs/
```

---

## 🤖 Automatic Mode

If you’re not sure what to configure:

* The system automatically selects the model, optimizer, precision, and training parameters.
* GPU/CPU are detected, and batch size and precision are optimized for available resources.

> ⚙️ Perfect for beginners: AutoTrainer Wizard can *self-configure* without requiring any technical expertise.

---

## 🧩 Roadmap

* Support for distributed training
* Integration with Weights & Biases
* Advanced logging and web dashboard
* Fine-tuning for multimodal models

---

## 🧑‍💻 Author

**MattimaxForce**
📦 GitHub: [MattimaxForce/AutoTrainer](https://github.com/MattimaxForce/AutoTrainer)

---

## 📜 License

This project is released under the **MIT License**.
