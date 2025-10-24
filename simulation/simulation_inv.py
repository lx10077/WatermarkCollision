import numpy as np
from tqdm import tqdm
import json
from scipy.stats import norm
from generator_inv import Inv_repeated_token_generator, generate_null_difs_with_copying
from score_inv import compute_scores_with_prefixes
from IPython import embed

K = 1000
c = 5
max_window=5
Delta = 0.7
Final_T = 700
key = 23333
alpha = 0.01
N_trial = 2000
top_prob = 0.9
generate_data =False

name = f"results_data/K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-inv{top_prob}"
print("Used Delta is:", Delta)

check_points = np.arange(1,1+Final_T)

def rowwise_unique_ratio_complement(matrix: np.ndarray) -> float:
    ratios = []
    for row in matrix:
        uniq_count = len(np.unique(row))
        total_count = len(row)
        ratios.append(uniq_count / total_count)
    mean_ratio = np.mean(ratios)
    return 1 - mean_ratio

def generate_watermark_text(prompt, T=60, c=5, Delta=0.5, key=1):
    gen = Inv_repeated_token_generator(vocab_size=K,max_window=max_window,Delta=Delta,c=c, key=key,p_types=(top_prob,(1-top_prob)/2,(1-top_prob)/2))
    outputs = gen(prompt.copy(), T)
    new_difs = outputs["new_difs"]
    new_Us = outputs["new_Us"]
    new_unique_difs, counts, groups = gen.summarize_prns(new_difs, new_Us)
    return new_difs[:T], new_unique_difs[:T], counts[:T], groups[:T]

if generate_data:
    prompts = []
    original_Ys = []
    unique_Ys = []
    new_Us = []
    counts = []
    groups = []

    for trial in tqdm(range(N_trial)):
        prompt = np.random.randint(K, size=c).tolist()
        Delta_ins = np.random.uniform(0.001, Delta)
        orinal_Y, unique_Y, count, group = generate_watermark_text(prompt, T=Final_T, c=c, key=key, Delta=Delta_ins)

        prompts.append(prompt)
        original_Ys.append(orinal_Y)
        unique_Ys.append(unique_Y)
        counts.append(count)
        groups.append(group)

    Null_Ys, unique_Null_Ys, null_count, null_group = generate_null_difs_with_copying(N_trial=N_trial*5, final_T=Final_T, K=K, p_types=(top_prob,(1-top_prob)/2,(1-top_prob)/2))

    save_dict = dict()
    save_dict["p"] = np.array(prompts).tolist()

    save_dict["o_y"] = np.array(original_Ys).tolist()
    save_dict["u_y"] = np.array(unique_Ys).tolist()
    save_dict["count"] = np.array(counts).tolist()
    save_dict["group"] = np.array(groups).tolist()

    save_dict["null_o_y"] = np.array(Null_Ys).tolist()
    save_dict["null_u_y"] = np.array(unique_Null_Ys).tolist()
    save_dict["null_count"] = np.array(null_count).tolist()
    save_dict["null_group"] = np.array(null_group).tolist()

    json.dump(save_dict, open(name+".json", 'w'))
else:
    save_dict = json.load(open(name+".json", "r"))

    prompts = save_dict["p"] 
    original_Ys = np.array(save_dict["o_y"])
    unique_Ys = np.array(save_dict["u_y"])
    counts = np.array(save_dict["count"])
    groups = np.array(save_dict["group"])

    print("Original dup ratio:", rowwise_unique_ratio_complement(original_Ys))
    print("New dup ratio:", np.mean(counts != 1))
    print()
    print("Alter mean:", original_Ys.mean())
    print("Alter unique mean:", unique_Ys.mean())
    print()
    Null_Ys = np.array(save_dict["null_o_y"])
    unique_Null_Ys = np.array(save_dict["null_u_y"])
    null_count = np.array(save_dict["null_count"])
    null_group = np.array(save_dict["null_group"])

    print("Null mean:", Null_Ys.mean())
    print("Null unique mean:", unique_Null_Ys.mean())    

def h_opt_dif(Ds, delta=0.1):
    Ds = np.array(Ds)
    final_Y = - np.array(Ds)
    final_Y = np.log(np.maximum(1+final_Y/(1-delta),1e-4)/np.maximum(1+final_Y,0))
    cumsum_Ys = np.cumsum(final_Y, axis=1)
    
    qs = []
    for _ in range(10):
        Null_Ys_U = np.random.uniform(size=(N_trial*2, Ds.shape[1]))
        Null_Ys_pi_s = np.random.randint(low=0, high=K, size=(N_trial*2, Ds.shape[1]))
        Null_etas = np.array(Null_Ys_pi_s)/(K-1)
        null_final_Y = -np.abs(Null_Ys_U-Null_etas)
        null_final_Y = np.log(np.maximum(1+null_final_Y/(1-delta),1e-4)/np.maximum(1+null_final_Y,0))
        null_cumsum_Ys = np.cumsum(null_final_Y, axis=1)
        h_ind_qs = np.quantile(null_cumsum_Ys, 1-alpha, axis=0)
        qs.append(h_ind_qs)
    h_ind_qs = np.mean(qs, axis=0)

    results = (cumsum_Ys >= h_ind_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)

def compute_ind_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)

def h_id_dif(Ds):
    mu_dif = -1/3
    var_dif = 1/6 - 1/9
    h_id_dif_qs = compute_ind_q(1-alpha, mu_dif, var_dif, check_points)

    Ds = -np.array(Ds)
    cumsum_Ds = np.cumsum(Ds, axis=1)

    results = (cumsum_Ds >= h_id_dif_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)

import numpy as np
from tqdm import tqdm

def h_repeated_opt(Ys, all_groups, delta=0.1, alpha=0.01):
    Ys = np.abs(Ys)
    all_groups = np.array(all_groups, dtype=object)  # allow ragged rows
    assert Ys.shape == all_groups.shape

    detection_rows = []
    min_len = None

    for i in tqdm(range(len(Ys))):
        # Score for this sample
        out = compute_scores_with_prefixes(
            Ys[i], all_groups[i], delta
        )
        score_Y_cumsum = out["group_block_scores_cumsum"]   

        # score_Y_cumsum = compute_scores_with_prefixes(
        #     Ys[i], all_groups[i], delta
        # )["group_block_scores_cumsum"]                    # shape (G_i,)

        # Null distribution for *this* group's structure
        null_Ys_cumsum = compute_scores_with_prefixes(
            unique_Null_Ys_for_quantile[:N_trial*2], all_groups[i], delta
        )["group_block_scores_cumsum"]                    # shape (R, G_i)

        critical_values = np.quantile(null_Ys_cumsum, 1 - alpha, axis=0)  # (G_i,)
        row_bool = (score_Y_cumsum >= critical_values)                     # (G_i,)

        detection_rows.append(row_bool)
        L = row_bool.shape[0]
        min_len = L if (min_len is None or L < min_len) else min_len

    if not detection_rows:
        return np.array([]), np.array([])

    # Truncate all rows to the minimal length and stack -> (num_samples, min_len)
    det_mat = np.vstack([row[:min_len] for row in detection_rows])

    det_mean = det_mat.mean(axis=0)
    det_std  = det_mat.std(axis=0)

    return det_mean, det_std


if generate_data:
    Null_Ys_U = np.random.uniform(size=(10000, Final_T))
    Null_Ys_pi_s = np.random.randint(low=0, high=K, size=(10000, Final_T))
    Null_etas = np.array(Null_Ys_pi_s)/(K-1)
    unique_Null_Ys_for_quantile = np.abs(Null_Ys_U-Null_etas)

    ##############################################
    ##
    ## Check the Type II errors
    ##
    ##############################################
    result_dict = dict()
    result_dict["rep-dif-opt-01"] = h_repeated_opt(unique_Ys, groups, delta=0.1, alpha=0.01)[0].tolist()
    result_dict["rep-dif-opt-001"] = h_repeated_opt(unique_Ys, groups, delta=0.01, alpha=0.01)[0].tolist()
    result_dict["dif-opt-01"] = (h_opt_dif(original_Ys, delta=0.1)[0].tolist(),h_opt_dif(unique_Ys, delta=0.1)[0].tolist())
    result_dict["dif-opt-001"] = (h_opt_dif(original_Ys, delta=0.01)[0].tolist(),h_opt_dif(unique_Ys, delta=0.01)[0].tolist())
    result_dict["dif-opt-0001"] = (h_opt_dif(original_Ys, delta=0.001)[0].tolist(),h_opt_dif(unique_Ys, delta=0.001)[0].tolist())
    result_dict["dif"] = (h_id_dif(original_Ys)[0].tolist(), h_id_dif(unique_Ys)[0].tolist())
    json.dump(result_dict, open(name+"-result"+".json", 'w'))

    ##############################################
    ##
    ## Check the Type I errors
    ##
    ###############################################
    null_dict = dict()
    null_dict["rep-dif-opt-01"] = h_repeated_opt(unique_Null_Ys, null_group, delta=0.1, alpha=0.01)[0].tolist()
    null_dict["rep-dif-opt-001"] = h_repeated_opt(unique_Null_Ys, null_group, delta=0.01, alpha=0.01)[0].tolist()
    null_dict["dif"] = (h_id_dif(Null_Ys)[0].tolist(),h_id_dif(unique_Null_Ys)[0].tolist())
    null_dict["dif-opt-01"] = (h_opt_dif(Null_Ys, delta=0.1)[0].tolist(),h_opt_dif(unique_Null_Ys, delta=0.1)[0].tolist()) 
    null_dict["dif-opt-001"] = (h_opt_dif(Null_Ys, delta=0.01)[0].tolist(),h_opt_dif(unique_Null_Ys, delta=0.01)[0].tolist()) 
    null_dict["dif-opt-0001"] = (h_opt_dif(Null_Ys, delta=0.001)[0].tolist(),h_opt_dif(unique_Null_Ys, delta=0.001)[0].tolist()) 
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
def labelize_inv(name):
    if name == "dif-opt-01":
        return r"$h_{\mathrm{dif},0.1}$"
    if name == "dif-opt-005":
        return r"$h_{\mathrm{dif},0.005}$"
    if name == "dif-opt-0001":
        return r"$h_{\mathrm{dif},0.001}$"
    if name == "dif-opt-001":
        return r"$h_{\mathrm{dif},0.01}$"
    
    if name == "rep-dif-opt-01":
        return r"$h_{\mathrm{new},0.1}$"
    if name == "rep-dif-opt-005":
        return r"$h_{\mathrm{new},0.005}$"
    if name == "rep-dif-opt-0001":
        return r"$h_{\mathrm{new},0.001}$"
    if name == "rep-dif-opt-001":
        return r"$h_{\mathrm{new},0.01}$"
    
    elif name == "id":
        return r"$h_{\mathrm{id}}$"
    elif name == "dif":
        return r"$h_{\mathrm{neg}}$"
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

for j, algo in enumerate(["dif", "dif-opt-01", "dif-opt-001", "rep-dif-opt-01", "rep-dif-opt-001"]):
    if "rep" not in algo:
        orignal, unique = null_dict[algo]
        x = np.arange(1, len(orignal)+1)
        ax[0].plot(x, np.array(orignal), linestyle=":",color=colors[j%len(colors)])
    else:
        unique = null_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[0].plot(x, unique, label=labelize_inv(algo), linestyle="-",color=colors[j%len(colors)])

ax[0].set_title(r"$H_0$")
ax[0].axhline(y=0.01, color="black", linestyle="dotted")
ax[0].set_ylabel(r"Type I error")
ax[0].set_xlabel(r"Unwatermarked text length")

for j, algo in enumerate(["dif", "dif-opt-01", "dif-opt-001", "rep-dif-opt-01", "rep-dif-opt-001"]):
    if "rep" not in algo:
        orignal, unique = result_dict[algo]
        x = np.arange(1, len(orignal)+1)
        ax[1].plot(x, 1-np.array(orignal), linestyle=":",color=colors[j%len(colors)])
    else:
        unique = result_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[1].plot(x, 1-np.array(unique), label=labelize_inv(algo), linestyle="-",color=colors[j%len(colors)])

ax[1].set_title(rf"$H_1, \Delta \sim U(0.001, {Delta})$")
ax[1].set_ylabel(r"Type II error")
ax[1].set_xlabel(r"Watermarked text length")
if use_log:
    ax[1].set_yscale('log')
ax[1].legend()

plt.tight_layout()
plt.savefig(f'K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-inv{top_prob}.pdf', dpi=300)
