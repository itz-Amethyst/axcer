# Axcer

<p align="center">
  <img
    src="https://i.ibb.co/nM8cZY4w/nano-banana-free-w-K3i-X60c9-Sqzg-Ml0-Z7-OIlg9q-Gs-PZ0mpak-Y4-removebg-preview-1.png"
    alt="Nano Banana Logo"
    width="250"
    height="240"
  />
</p>


<!-- ![Build Status](https://img.shields.io/github/actions/workflow/status/itz-Amethyst/axcer/python-app.yml?branch=main)   -->
<!-- ![PyPI Version](https://img.shields.io/pypi/v/axcer)   -->
![License](https://img.shields.io/badge/license-AGPL--3.0-red)

**Axcer** (<u>A</u>lgorithmic compression with le<u>X</u>ical and interrogative <u>E</u>xt<u>R</u>action) is a *training‑free, GPU‑independent prompt compression framework* designed for modern NLP workflows with large language models (LLMs). It extracts lexical units and interrogative cues to compress prompts efficiently, enabling faster and more accurate downstream inference without the overhead of training data or heavy models.

---

## 📌 Overview

Prompt compression relying on keyword extraction is a prominent direction in Natural Language Processing (NLP) due to the increasing demand for processing larger input sequences by LLMs. Existing compression methods often depend on **trained masked‑language models (MLMs)** and incur **high latency** or **GPU requirements**, limiting their practical use.

Axcer addresses these challenges by:

- Being **training‑free** and **hardware‑agnostic** (no GPU needed) ⚡
- Extracting meaningful lexical units and interrogative words crucial for QA and other tasks
- Constructing a **Weighted Directed Graph (WDG)** to capture structural semantics of input prompts
- Achieving **~33× faster compression** 🚀 while outperforming baselines in accuracy across multiple tasks

> Unlike other prompt compression algorithms such as **LLMLingua** and **Selective Context**, Axcer works **independently of LLMs** and does not require training data, making it lightweight and resource-friendly.

This software project accompanies the research paper published in WILL BE UPDATED! ([]())

---

## 📁 Repository Structure

```

axcer/                   # Main source code for the Axcer framework
experiments/             # All experimental artifacts
├─ modals/               # Inference experiments on specific model code
├─ results/              # Results organized by baseline and LLM
├─ evaluation/           # Scripts for evaluation metrics
│   └─ visualization/    # Plotting/visualization scripts and figures
datasets/                # 9 benchmark datasets used for experiments
│   ├─ SQuAD
│   ├─ BoolQ
│   ├─ PIQA
│   ├─ AI2_ARC
│   ├─ MRPC
│   ├─ GSM8K
│   ├─ MAWPS
│   ├─ SciTLDR
│   └─ MBPP
figures/                  # Figures used in the manuscript

````

---

## 📦 Installation

Axcer uses a modern Python packaging layout (`pyproject.toml`). To install:

```bash
# pip way
pip install axcer

# uv way
uv pip install axcer
````

> In order to construct weighted directed graph (WDG), first install optional visualization dependencies (e.g., PyQt5):
>
> ```bash
> pip install axcer[vis]
> ```


---

## 🚀 Usage Example: Compressing Prompts

Axcer is flexible and can compress both single prompts and a batch of prompts at once.

```python
from axcer import Axcer

# Initialize the compressor
a = Axcer()

# Compress a single prompt
prompt = """If Sam and Harry have 100 feet of fence between them, and they agree to split it
with Harry getting 60 feet more than Sam, how much is left over for Sam?"""
compressed_prompt = a.compress_prompt(prompt)
print("Compressed single prompt:", compressed_prompt)

# Compress a list of prompts
prompts = [
    """Janet buys a brooch for her daughter.  She pays $500 for the material to make it
    and then another $800 for the jeweler to construct it.  After that, she pays 10% of
    that to get it insured.  How much did she pay?""",
    .
    .
    .
]
compressed_prompts = a.compress_prompt(prompts)
print("Compressed batch prompts:", compressed_prompts)

```

In order to see visualization graph, call the visualiztion function like this:

```python

a.graph_processor.show_graph()
```

### Runtime & Memory Logging

Axcer can track performance metrics if `enable_metrics_logging=True`. Metrics recorded:

* **total_runtime**: compression time in seconds
* **total_memory_kb**: memory allocated during compression (Python allocations)

Manual early‑stop hooks:

```python
if self.enable_metrics_logging:
    self._set_runtime_now()
    self._set_memory_now()
```

---

## 🧪 Experiments

All experiments are located in `experiments/`:

* **modals/**: Inference experiments on LLMs
* **results/**: Structured results by baseline and LLM
* **evaluation/**: Evaluation metrics and analysis scripts
* **visualization/**: Plotting scripts and figures using Matplotlib/Seaborn

Datasets used in experiments are under `datasets/`.

---

## 📊 Visualization & Figures

* **visualization/**: Scripts for generating plots
* **figures/**: High-resolution figures used in the manuscript

---

## 📚 Datasets

Axcer was evaluated on **nine benchmark datasets**:

* **SQuAD**: QA
* **BoolQ**, **MRPC**: Binary classification
* **PIQA**: Commonsense reasoning
* **AI2_ARC**: Multiple choice reasoning
* **GSM8K**, **MAWPS**: Math reasoning
* **SciTLDR**: Summarization
* **MBPP**: Python code compression

---

## 📜 Citation

```
will be added
```

---

## 🛡️ License

Axcer is licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html)**.

* You **may use, modify, and redistribute** this software.
* Any redistribution or derivative work **must also be licensed under AGPL-3.0**.
* Private proprietary forks, renaming, or commercial redistribution without compliance is **prohibited**.

  <!-- [![Code Quality](https://github.com/lmcache/lmcache/actions/workflows/code_quality_checks.yml/badge.svg?branch=dev&label=tests)](https://github.com/LMCache/LMCache/actions/workflows/code_quality_checks.yml) -->
