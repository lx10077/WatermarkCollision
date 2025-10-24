# Optimal Detection for Language Watermarks with Pseudorandom Collision – LLM Code

This file implements the **empirical evaluation pipeline for watermark detection under pseudorandom collision** using large language models. It complements the simulation study by operating on LLM-generated text under two watermarking schemes:
- **Gumbel-Max watermark**
- **Inverse-Transform watermark**

The workflow consists of three major stages:

1. ⚙️ **Data Generation** – create watermarked and null text samples.  
2. 📊 **Probability Recovery** – recompute token-level probabilities from the same model.  
3. 🧪 **Detection & Evaluation** – run statistical detection and visualize results.

---

## ⚙️ Data Generation

### 📂 Files
| File | Description |
|------|--------------|
| `prepare_data_gum.py` | Generate **Gumbel-Max watermarked** samples. |
| `prepare_data_inv.py` | Generate **Inverse-Transform watermarked** samples. |
| `prepare_data_null.py` | Generate **null (human-written)** samples for baseline testing. |
| `generation.py`, `sampling.py` | Provide shared generation logic and sampling methods. |

Each script:
1. Uses the specified model (e.g., `facebook/opt-1.3b`) and temperature setting.  
2. Generates sequences up to **800 tokens** per prompt.  
3. Ensures that at least **300 unique pivotal statistics** are obtained under the same pseudo-random collision; otherwise, the generation is skipped.  
4. Saves resulting text data into `text_data/` with filenames encoding model, (watermarking) method, context size *c*, sequence length *m*, and temperature.

### ▶️ Example
```bash
python prepare_data_gum.py --model facebook/opt-1.3b --temperature 0.3
python prepare_data_inv.py --model facebook/opt-1.3b --temperature 0.5
python prepare_data_null.py --model facebook/opt-1.3b
````
Each call will generate watermarked or extract human-written datasets satisfying the unique pivotal-statistic requirement, ready for the recovery and detection stages.
All default parameters are defined inside the scripts, ensuring reproducible setups.

---

## 📊 Probability Recovery

### 📂 Files

| File                 | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `recover_Ps.py`      | Recover top-token probabilities **for both Gumbel and Inverse watermark data**. |

### ▶️ Usage

Simply run:

```bash
python recover_Ps.py --model facebook/opt-1.3b --temperature 0.3 --method Gumbel   # For Gumbel
python recover_Ps.py --model facebook/opt-1.3b --temperature 0.5 --method Inverse  # For Inverse

```

This will automatically:

* Load all existing datasets from `text_data/`,
* Apply the specified model and temperature,
* Compute and save top-token probabilities to `recoverPs/` or `null_recoverPs/` for each token.

Each call will produce two output files: one for the watermarked data and one for the null data.

Output files:
```
recoverPs/
 ├── 1p3B-Gumbel-c4-m500-T2000-noncomm_prf-15485863-temp0.3-stream-unique-raw-Ps-all.pkl
 └── 1p3B-Inverse-c4-m500-T2000-noncomm_prf-15485863-temp0.5-stream-unique-raw-Ps-all.pkl

null_recoverPs/
 ├── 1p3B-Gumbel-c4-m500-T2000-noncomm_prf-15485863-temp0.3-null-Ps-all.pkl
 └── 1p3B-Inverse-c4-m500-T2000-noncomm_prf-15485863-temp0.5-null-Ps-all.pkl

```

---

## 🧪 Detection and Evaluation

### 📂 Files

| File                  | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `detect_gum.py` | Performs statistical detection on **Gumbel-Max watermarked data**. Evaluates all detectors (Log, ARS, OptGum, TrGoF, Switch, RepeatWeight). |
| `detect_inv.py` | Performs statistical detection on **Inverse-Transform watermarked data**. Evaluates detectors (OptInv, IdInv, RepeatOptInv).                |
| `detectors.py`        | Contains all statistical detectors (e.g., OptGum, OptInv, Id, Repeat variants). |               |
| `score_inv_inhomo.py` | Defines scoring functions for inverse-transform detectors.                      |

### ▶️ Example

```bash
python detect_gum.py --model facebook/opt-1.3b --method Gumbel --temperature 0.3
python detect_inv.py --model facebook/opt-1.3b --method Inverse --temperature 0.5
```

Each detection script:

* Loads recovered probabilities and null references,
* Computes test statistics across samples,
* Evaluates Type I / Type II error rates,
* Plots and saves the detection performance figure to `plot/`.

Output examples:

```
plot_data/
 ├── 1p3B-Gumbel-c4-m800-noncomm_prf-15485863-temp0.3-stream-unique-300-all_scores.pkl
 └── 1p3B-Inverse-c4-m800-noncomm_prf-15485863-temp0.5-stream-unique-300-all_scores.pkl
```

Each `.pkl` file contains all detector outputs, including Type I/II error arrays for each detector (e.g., Log, ARS, Opt, GoF, Oracle, Switch). 

Plots are **always saved automatically**, which visualize the Type I error (false positive rate) and Type II error (false negative rate) across text lengths for each detector.


## 🧠 Quick Start

### For Gumbel-Max watermark

```bash
python prepare_data_gum.py
python prepare_data_null.py
python recover_Ps.py
python detect_gum.py
```

### For Inverse-Transform watermark

```bash
python prepare_data_inv.py
python prepare_data_null.py
python recover_Ps.py
python detect_inv.py
```

All intermediate results (token data, recovered probabilities, plots) will be saved automatically in:

```
text_data/      # generated samples
recoverPs/      # recovered probabilities
plot/           # evaluation figures
```
