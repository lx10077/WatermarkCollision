#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import json
from IPython import embed
from scipy.stats import gamma, norm, binom
from generator_gum import Gumbel_repeated_token_generator, generate_null_with_copying

K = 1000
c = 7
max_window=5
Delta = 0.1
Final_T = 700
key = 23333
alpha = 0.01
N_trial = 2000
top_prob = 0.9

name = f"results_data/K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-max{top_prob}"
generate_data = True

def rowwise_unique_ratio_complement(matrix: np.ndarray) -> float:
    ratios = []
    for row in matrix:
        uniq_count = len(np.unique(row))
        total_count = len(row)
        ratios.append(uniq_count / total_count)
    mean_ratio = np.mean(ratios)
    return 1 - mean_ratio

print("Used Delta is:", Delta)

## CDF and PDF
def F(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += Probs[k]*x**(1/Probs[k])
    return rho

def f(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += x**(1/Probs[k]-1)
    return rho


## Compute critial values
check_points = np.arange(1,1+Final_T)

def compute_gamma_q(q, check_point):
    qs = []
    for t in check_point:
        qs.append(gamma.ppf(q=q,a=t))
    return np.array(qs)

def compute_ind_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)

h_ars_qs = compute_gamma_q(1-alpha, check_points)
h_log_qs = compute_gamma_q(alpha, check_points)

def generate_watermark_text(prompt, T=60, c=4, Delta=0.5, key=1):
    gen = Gumbel_repeated_token_generator(vocab_size=K,max_window=max_window,Delta=Delta,c=c, key=key,p_types=(top_prob,(1-top_prob)/2,(1-top_prob)/2))
    outputs = gen(prompt.copy(), T)
    new_xis = outputs["new_selected_xis"]
    new_unique_xis, counts = gen.summarize_prns(new_xis)
    return new_xis[:T], new_unique_xis[:T], counts[:T]
    
if generate_data:
    prompts = []
    original_Ys = []
    unique_Ys = []
    counts = []

    for trial in tqdm(range(N_trial)):
        prompt = np.random.randint(K, size=c).tolist()
        Delta_ins = np.random.uniform(0.001, Delta)
        orinal_Y, unique_Y, count = generate_watermark_text(prompt, T=Final_T, c=c, key=key, Delta=Delta_ins)

        prompts.append(prompt)
        original_Ys.append(orinal_Y)
        unique_Ys.append(unique_Y)
        counts.append(count)

    Null_Ys, unique_Null_Ys, null_count = generate_null_with_copying(N_trial=10**4, final_T=Final_T, p_types=(top_prob,(1-top_prob)/2,(1-top_prob)/2))

    save_dict = dict()
    save_dict["p"] = np.array(prompts).tolist()

    save_dict["o_y"] = np.array(original_Ys).tolist()
    save_dict["u_y"] = np.array(unique_Ys).tolist()
    save_dict["count"] = np.array(counts).tolist()

    save_dict["null_o_y"] = np.array(Null_Ys).tolist()
    save_dict["null_u_y"] = np.array(unique_Null_Ys).tolist()
    save_dict["null_count"] = np.array(null_count).tolist()

    json.dump(save_dict, open(name+".json", 'w'))
else:
    save_dict = json.load(open(name+".json", "r"))

    prompts = save_dict["p"] 
    original_Ys = np.array(save_dict["o_y"])
    unique_Ys = np.array(save_dict["u_y"])
    counts = np.array(save_dict["count"])

    print("Original dup ratio:", rowwise_unique_ratio_complement(original_Ys))
    print("New dup ratio:", np.mean(counts != 1))
    print()
    print("Alter mean:", original_Ys.mean())
    print("Alter unique mean:", unique_Ys.mean())
    print()
    Null_Ys = np.array(save_dict["null_o_y"])
    unique_Null_Ys = np.array(save_dict["null_u_y"])
    null_count = np.array(save_dict["null_count"])
    print("Null mean:", Null_Ys.mean())
    print("Null unique mean:", unique_Null_Ys.mean())

def h_ars(Ys):
    Ys = np.array(Ys)
    h_ars_Ys = -np.log(1-Ys)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_ars_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)

def h_opt(Ys, delta0=0.2):
    # This is for the optimal score function
    Ys = np.array(Ys)

    def f(r, delta):
        inte_here = np.floor(1/(1-delta))
        rest = 1-(1-delta)*inte_here
        return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1))
    
    h_ars_Ys = f(Ys, delta0)
    collection = []
    for _ in range(10):
        Null_Ys = np.random.uniform(size=(N_trial*2, Final_T))
        Simu_Y = f(Null_Ys, delta0)
        Simu_Y = np.cumsum(Simu_Y, axis=1)
        h_help_qs = np.quantile(Simu_Y, 1-alpha, axis=0)
        collection.append(h_help_qs)
    h_help_qs = np.mean(collection, axis=0)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_help_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)

def h_log(Ys):
    Ys = np.array(Ys)
    h_log_Ys = np.log(Ys)
    cumsum_Ys = np.cumsum(h_log_Ys, axis=1)
    
    results = (cumsum_Ys >= -h_log_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)

def h_ind(Ys, ind_delta=0.5):
    Ys = np.array(Ys)
    h_ind_Ys = (Ys >= ind_delta)
    cumsum_Ys = np.cumsum(h_ind_Ys, axis=1)
    h_ind_qs = binom.ppf(n=check_points, p = 1-ind_delta, q = 1-alpha)
    results = (cumsum_Ys >= h_ind_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)

Delta1Star = np.array([0,0,0.29707273,0.38798173,0.42366049,0.44200376,0.45318023,0.46069369,0.46610054,0.47020137])

# Uses global 1-D array Delta1Star (same as your original code).
def h_repeated_opt(Ys, all_counts, delta=0.1, alpha=0.01, eps=1e-6):
    """
    One-batch Monte Carlo critical values (no reps loop), memory-lean.
    Returns:
        det_mean: (T,) mean detection rate across rows
        det_std:  (T,) std of detection indicator across rows
    """
    Ys = np.array(Ys)
    all_counts = np.array(all_counts)
    assert Ys.shape == all_counts.shape
    N_trial, T = Ys.shape
    L = int(len(Delta1Star))

    # ---- constants kept internal (no new args) ----
    null_factor = 2          # size of null batch ≈ 2 * N_trial
    dtype = np.float32

    # ---------- observed scores (vectorized) ----------
    p = 1.0 - delta
    p_safe = np.clip(p, eps, 1.0 - eps)
    w1 = dtype(1.0 / p_safe - 1.0)
    w2 = dtype(1.0 / (1.0 - p_safe + 1e-6) - 1.0)

    Ysafe = Ys.astype(dtype, copy=True)
    np.clip(Ysafe, eps, 1.0, out=Ysafe)
    ylog = np.log(Ysafe)

    s2_obs = ylog * w1
    tmp1 = np.exp(ylog * w1)
    tmp2 = np.exp(ylog * w2)
    tmp1 += tmp2
    tmp1 += dtype(1e-6)
    s1_obs = np.log(tmp1)
    del tmp2, tmp1, ylog, Ysafe

    if L > 0:
        idx = np.clip(all_counts - 1, 0, L - 1)
        thresh = np.take(Delta1Star, idx, mode='clip').astype(dtype, copy=False)
        cond = (all_counts >= 1) & (all_counts <= L) & (dtype(delta) >= thresh)
    else:
        cond = np.zeros_like(all_counts, dtype=bool)

    score_obs = s2_obs.copy()
    np.copyto(score_obs, s1_obs, where=cond)
    del s1_obs, s2_obs
    np.cumsum(score_obs, axis=1, out=score_obs)  # cumsum of observed scores

    # ---------- single-batch Monte Carlo for criticals ----------
    rng = np.random.default_rng(None)
    R = max(null_factor * N_trial, 1)            # null batch rows
    k = int(np.ceil((1.0 - alpha) * R) - 1)      # order-stat index for (1-alpha) quantile
    k = min(max(k, 0), R - 1)

    # U -> log(U) in place (R,T)
    U = rng.uniform(size=(R, T)).astype(dtype, copy=False)
    np.clip(U, eps, 1.0, out=U)
    np.log(U, out=U)

    # s2 = log(U)*w1 ; buf = s1 - s2
    s2 = U * w1
    buf = np.exp(U * w1)
    tmp = np.exp(U * w2)
    buf += tmp
    buf += dtype(1e-6)
    np.log(buf, out=buf)      # buf = s1_null
    del tmp, U
    buf -= s2                 # buf = (s1_null - s2_null)

    # prefix sums of s2
    np.cumsum(s2, axis=1, out=s2)  # s2 = cumsum_s2

    # workspace (R,T) reused per row
    work = np.empty_like(buf)
    criticals = np.empty((N_trial, T), dtype=dtype)

    for j in tqdm(range(N_trial)):
        # work = cumsum( buf * cond[j] ) + cumsum_s2
        np.multiply(buf, cond[j], out=work)   # broadcast (T,) -> (R,T)
        np.cumsum(work, axis=1, out=work)
        work += s2

        # in-place selection for (1-alpha) quantile along axis=0
        work.partition(k, axis=0)             # no full sort, low memory
        criticals[j] = work[k, :].astype(dtype, copy=False)

    del s2, buf, work

    # ---------- detection summary ----------
    det_sum = np.zeros(T, dtype=np.int64)
    for j in range(N_trial):
        det_sum += (score_obs[j] >= criticals[j])

    det_mean = det_sum.astype(np.float64) / float(N_trial)
    det_std = np.sqrt(det_mean * (1.0 - det_mean))

    return det_mean, det_std

##############################################
##
## Check the Type II errors
##
##############################################
if generate_data:
    result_dict = dict()
    
    result_dict["ars"] = (h_ars(original_Ys)[0].tolist(),h_ars(unique_Ys)[0].tolist())
    result_dict["log"] = (h_log(original_Ys)[0].tolist(),h_log(unique_Ys)[0].tolist())
    result_dict["rep-opt-001"] = h_repeated_opt(unique_Ys,counts,0.01)[0].tolist()
    result_dict["rep-opt-0005"] = h_repeated_opt(unique_Ys,counts,0.005)[0].tolist()
    result_dict["opt-0001"] = (h_opt(original_Ys,0.001)[0].tolist(),h_opt(unique_Ys,0.001)[0].tolist())
    result_dict["opt-0005"] = (h_opt(original_Ys,0.005)[0].tolist(),h_opt(unique_Ys,0.005)[0].tolist())
    result_dict["opt-001"] = (h_opt(original_Ys,0.01)[0].tolist(),h_opt(unique_Ys,0.01)[0].tolist())
    result_dict["opt-01"] = (h_opt(original_Ys,0.1)[0].tolist(),h_opt(unique_Ys,0.1)[0].tolist())

    json.dump(result_dict, open(name+"-result.json", 'w'))

    ##############################################
    ##
    ## Check the Type I errors
    ##
    ##############################################
    null_dict = dict()

    null_dict["ars"] =(h_ars(Null_Ys)[0].tolist(),h_ars(unique_Null_Ys)[0].tolist())
    null_dict["log"] = (h_log(Null_Ys)[0].tolist(),h_log(unique_Null_Ys)[0].tolist())
    null_dict["rep-opt-0005"] = h_repeated_opt(unique_Null_Ys,null_count,0.005)[0].tolist()
    null_dict["rep-opt-001"] = h_repeated_opt(unique_Null_Ys,null_count,0.01)[0].tolist()
    null_dict["opt-01"] = (h_opt(Null_Ys, 0.1)[0].tolist(),h_opt(unique_Null_Ys, 0.1)[0].tolist())
    null_dict["opt-001"] = (h_opt(Null_Ys, 0.01)[0].tolist(),h_opt(unique_Null_Ys, 0.01)[0].tolist())
    null_dict["opt-0005"] = (h_opt(Null_Ys, 0.005)[0].tolist(),h_opt(unique_Null_Ys, 0.005)[0].tolist())
    null_dict["opt-0001"] = (h_opt(Null_Ys, 0.001)[0].tolist(),h_opt(unique_Null_Ys, 0.001)[0].tolist())

    json.dump(null_dict, open(name+"-null.json", 'w'))
else:
    with open(name+"-result.json", "r") as f:
        result_dict = json.load(f)
    with open(name+"-null.json", "r") as f:
        null_dict = json.load(f)

##############################################
##
## Plot
##
##############################################
def labelize(name):
    if name == "ars":
        return r"$h_{\mathrm{ars}}$"
    elif name == "log":
        return r"$h_{\mathrm{log}}$"
    if name == "opt-01":
        return r"$h_{\mathrm{gum},0.1}$"
    if name == "opt-001":
        return r"$h_{\mathrm{gum},0.01}$"
    if name == "opt-0001":
        return r"$h_{\mathrm{gum},0.001}$"
    if name == "opt-0005":
        return r"$h_{\mathrm{gum},0.005}$"
    if name == "rep-opt-0005":
        return r"$h_{\mathrm{new},0.005}$"
    if name == "rep-opt-001":
        return r"$h_{\mathrm{new},0.01}$"
    else:
        raise KeyError(f"{name}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'lines.linewidth': 1,
    'font.size': 13,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} '
})

use_log = True
first = 700
linestyles = ["-", "-.", ":", "--", "-."]
colors = ["tab:blue", "tab:orange", "tab:brown", "tab:red", "tab:gray", "black",  "tab:purple",  "tab:pink",]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

for j, algo in enumerate(["ars", "log", "opt-0005", "rep-opt-0005"]):
    if "rep" not in algo:
        orignal, unique = null_dict[algo]
        x = np.arange(1, len(orignal)+1)
        ax[0].plot(x, np.array(orignal), linestyle=":",color=colors[j%len(colors)])
    else:
        unique = null_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[0].plot(x, unique, label=labelize(algo), linestyle="-",color=colors[j%len(colors)])

ax[0].set_title(r"$H_0$")
ax[0].axhline(y=0.01, color="black", linestyle="dotted")
ax[0].set_ylabel(r"Type I error")
ax[0].set_xlabel(r"Unwatermarked text length")

for j, algo in enumerate(["ars", "log",  "opt-0005", "rep-opt-0005"]):
    if "rep" not in algo:
        orignal, unique = result_dict[algo]
        x = np.arange(1, len(orignal)+1)
        ax[1].plot(x, 1-np.array(orignal), linestyle=":",color=colors[j%len(colors)])
    else:
        unique = result_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[1].plot(x, 1-np.array(unique), label=labelize(algo), linestyle="-",color=colors[j%len(colors)])

ax[1].set_title(rf"$H_1, \Delta \sim U(0.001, {Delta})$")
ax[1].set_ylabel(r"Type II error")
ax[1].set_xlabel(r"Watermarked text length")
if use_log:
    ax[1].set_yscale('log')
ax[1].legend()

plt.tight_layout()
plt.savefig(f'K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-max{top_prob}.pdf', dpi=300)
