# Layer-wise Calibrated Quantization for Transformers

## 📌 Overview

This project implements a research-inspired post-training quantization (PTQ) pipeline for transformer models (DistilGPT-2). It explores how layer sensitivity impacts quantization performance and demonstrates stable mixed-precision quantization using calibration.

---

## ⚙️ Key Features

* Layer-wise sensitivity analysis (DeltaLoss)
* Per-tensor scale optimization
* Calibration-based quantization
* Mixed precision (8-bit / 6-bit)
* Visualization of sensitivity and error propagation

---

## 📊 Results

| Model                      | Perplexity |
| -------------------------- | ---------- |
| FP32                       | ~94        |
| 8-bit                      | ~102       |
| Mixed (8/6-bit calibrated) | ~132       |

---

## 📈 Visualizations

* Layer-wise sensitivity heatmap
* Quantization error across layers
* Perplexity vs bit-width tradeoff

---

## 🧠 Key Insight

Low-bit quantization in transformers is highly sensitive to scaling and activation distribution. Calibration is essential to maintain stability.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

Run notebook:

```bash
jupyter notebook notebook/quantization.ipynb
```

---

## 📚 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Matplotlib

---

## 👤 Author

**Ganta Parthiv Kumar**
