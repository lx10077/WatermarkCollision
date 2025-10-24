# 🔍 Optimal Detection for Language Watermarks under Pseudorandom Collision

This repository provides the full experimental framework for **detecting statistical language watermarks under pseudo-random collisions**.  
It contains two main components:

---

## 📁 `simulation/`

Implements **controlled Monte Carlo simulations** for validating theoretical results and comparing statistical detectors.

**Key files:**
- `simulation_gum.py` – Simulation of detection for **Gumbel-Max watermark**.
- `simulation_inv.py` – Simulation of detection for **Inverse-Transform watermark**.
- `detectors.py` – Core statistical detectors (shared across all experiments).
- `generator_gum.py`, `generator_inv.py` – Generate synthetic watermark and null samples.
- `plot.py` – Visualize empirical Type I/II errors.

➡️ *For details on parameters and running instructions, see* [`simulation/README.md`](simulation/README.md).

---

## 📁 `LLM/`

Implements the **empirical evaluation pipeline** using **large language models (LLMs)** to generate, recover, and detect watermark signals under pseudo-random collision.

**Key files:**
- `prepare_data_gum.py` – Generate Gumbel-Max watermarked text.
- `prepare_data_inv.py` – Generate Inverse-Transform watermarked text.
- `prepare_data_null.py` – Generate null (unwatermarked) text.
- `recover_Ps.py` – Recompute top-token probabilities from generated sequences.
- `detect_gum.py` – Apply all detectors to Gumbel-Max data.
- `detect_inv.py` – Apply all detectors to Inverse-Transform data.
- `detectors.py` – Statistical detectors used for both watermark types.

➡️ *For detailed workflow and hyperparameter explanations, see* [`LLM/README.md`](LLM/README.md).

---

## 🧠 Summary

Both modules study **watermark detection under pseudo-random collision**, where model sampling and pseudo-random functions are synchronized.  The `simulation` part provides controlled validation, while `LLM` replicates the same logic on **real model-generated text** for empirical verification.

---

✅ *For reproduction or further exploration, please consult the README files within each folder.*
