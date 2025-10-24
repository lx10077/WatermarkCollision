# Optimal Detection for Language Watermarks with Pseudorandom Collision – Simulation Code

This repository contains simulation and analysis scripts for evaluating **optimal detection rules under pseudorandom collisions**.  
The experiments deliberately introduce repetition to study how such collisions affect **Type I** and **Type II** error control in watermark detection.

The simulations compare two types◊ of watermark generation processes:
- **Gumbel-max watermarking**
- **Inverse-transform watermarking**

Each main script generates synthetic data, computes detection statistics, and produces plots comparing detection power and Type I error rates across methods.

---

## ⚙️ Simulation Setup

We deliberately introduce **repetition** to evaluate Type I and Type II errors under pseudorandom collisions.  
The vocabulary size is set to $|\mathcal{V}| = 10^3$.  
At each time step $t$:

- With **probability 0.9**, a **new token** is generated according to the chosen watermarking scheme.  
  Specifically:
  1. Sample $\Delta_t \sim \mathrm{Unif}(10^{-3}, \Delta_{\max})$ for a prespecified $\Delta_{\max} \in (0,1)$.
  2. Construct an NTP distribution $\mathbf{P}_t$ satisfying $\max_{v \in \mathcal{V}} \mathbf{P}_{t,v} = 1 - \Delta_t$.  
     The NTP distribution interpolates between a **Zipf law** and the **uniform distribution**, with $\Delta_{\max}$ controlling its entropy or randomness.

- With the remaining **probability 0.1**, repetition is introduced through two independent mechanisms:
  - **Stored-segment recall (0.05):** Insert a segment sampled from a growing pool of previously generated segments. The pool is updated whenever new segments appear.
  - **Prefix-copying (0.05):** Copy a contiguous block from the previously generated prefix.  
    A block length $L \in \{1, \dots, L_{\max}\}$ is drawn uniformly with $L_{\max}=5$, and a valid start index is selected uniformly.

The repetition decision is independent of the specific corruption mechanism, resulting in a **decoupled corruption** setup.  
This allows clean comparisons of Type I and II errors across different score functions and detection statistics.

---

## 📂 File Overview

### **1. `simulation_gum.py` (Main script – Gumbel-max)**
Runs Monte Carlo simulations under the **Gumbel-max watermark** model.  
Synthetic sequences are generated using the `Gumbel_repeated_token_generator`, and detection rules are evaluated for accuracy and power.

**Outputs:**
- `...-result.json` — results under the alternative hypothesis ($H_1$)
- `...-null.json` — results under the null hypothesis ($H_0$)

**Main hyperparameters:**

| Parameter | Description | Default |
|------------|--------------|----------|
| `K` | Vocabulary size | 1000 |
| `c` | Context window size for RNG seeding | 7 |
| `max_window` | Maximum copied interval length $L_{\max}$ | 5 |
| `Delta` | Upper bound $\Delta_{\max}$ controlling randomness | 0.7 |
| `Final_T` | Sequence length | 700 |
| `key` | Random seed | 23333 |
| `alpha` | Significance level | 0.01 |
| `N_trial` | Number of Monte Carlo trials | 2000 |
| `top_prob` | Probability (0.9) of generating a **new text window**, i.e., not copying from history | 0.9 |

**Usage:**
```bash
python simulation_gum.py
```
Set `generate_data=True` in the script to regenerate data.

---

### **2. `simulation_inv.py` (Main script – Inverse-transform)**
Runs the same experiment under the **inverse-transform watermark** model.  
Sequences are generated via the `Inv_repeated_token_generator`, and multiple score functions (`h_opt_dif`, `h_id_dif`, `h_repeated_opt`) are compared for detection efficiency.

**Usage:**
```bash
python simulation_inv.py
```

---

### **3. `generator_gum.py`**
Implements the generator for the Gumbel-max model.  
It alternates between:
1. Sampling a new token (with probability `top_prob` = 0.9),  
2. Copying a random prefix segment, and  
3. Inserting a stored segment from a growing memory pool.

This produces pseudorandom collisions consistent with the simulation framework.

---

### **4. `generator_inv.py`**
Defines the generator for the inverse-transform model, following the same repetition logic but based on inverse-transform sampling.

---

### **5. `score_inv.py`**
Implements the function `compute_scores_with_prefixes`, used to compute prefix-based statistics for the inverse-transform simulations.

---

### **6. `plot.py`**
Provides visualization utilities for comparing Type I and II errors of different detection methods.  
It loads JSON files produced by the simulation scripts and generates comparative plots.

**Example usage:**
```bash
python plot.py
```

---

## 🧪 Typical Workflow

1. **Run simulations**
   ```bash
   python simulation_gum.py
   # or
   python simulation_inv.py
   ```

2. **Generate plots**
   ```bash
   python plot.py
   ```

3. **Inspect results**
   - JSON summaries are saved in `results_data/`
   - Figures visualize detection performance under both $H_0$ and $H_1$

---

## 📊 Output Structure

```
results_data/
  ├── K1000N2000c7key23333T700Delta0.7-alpha0.01-max0.9-result.json
  ├── K1000N2000c7key23333T700Delta0.7-alpha0.01-max0.9-null.json
  └── ...
```

Each JSON file stores per-trial statistics for empirical Type I/II error estimation.

---

## 🧩 Dependencies

Install required packages before running:
```bash
pip install numpy scipy matplotlib tqdm
```

LaTeX support (optional) may be needed for rendering figure labels.

---

## 🧠 Summary

| File | Description |
|------|--------------|
| `simulation_gum.py` | Main simulation for Gumbel-max model |
| `simulation_inv.py` | Main simulation for inverse-transform model |
| `generator_gum.py` | Generator implementing repetition and Gumbel sampling |
| `generator_inv.py` | Generator implementing repetition and inverse-transform sampling |
| `score_inv.py` | Score computation for inverse-transform model |
| `plot.py` | Plotting script for visualizing results |

---
