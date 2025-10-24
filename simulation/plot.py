#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'lines.linewidth': 1.5,
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} '
})


use_log = True
def labelize(name):
    if name == "ars":
        return r"$h_{\mathrm{ars}}$"
    elif name == "log":
        return r"$h_{\mathrm{log}}$"
    
    if name == "opt-02":
        return r"$h_{\mathrm{gum},0.2}$"
    if name == "opt-005":
        return r"$h_{\mathrm{gum},0.05}$"
    if name == "opt-0005":
        return r"$h_{\mathrm{gum},0.005}$"
    if name == "opt-0001":
        return r"$h_{\mathrm{gum},0.001}$"
    if name == "opt-01":
        return r"$h_{\mathrm{gum},0.1}$"
    if name == "opt-001":
        return r"$h_{\mathrm{gum},0.01}$"

    if name == "rep-opt-02":
        return r"$h_{\mathrm{new},0.2}$"
    if name == "rep-opt-005":
        return r"$h_{\mathrm{new},0.05}$"
    if name == "rep-opt-0005":
        return r"$h_{\mathrm{new},0.005}$"
    if name == "rep-opt-0001":
        return r"$h_{\mathrm{new},0.001}$"
    if name == "rep-opt-01":
        return r"$h_{\mathrm{new},0.1}$"
    if name == "rep-opt-001":
        return r"$h_{\mathrm{new},0.01}$"
    else:
        raise KeyError(f"{name}")

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
    

K = 1000
c = 7
max_window=5
Delta = 0.7
Final_T = 700
key = 23333
alpha = 0.01
N_trial = 2000
top_prob = 9/10

colors = [
    "#6baed6",  # Group A 浅蓝
    "#ffbb78",  # Group B 浅橙黄
    "#1f77b4",  # Group A 深蓝
    "#ff7f0e",  # Group B 橙黄
    "#d62728",  # Group C 红色 (Ours)
    "#2ca02c",  # Group D 深绿
    "#98df8a",  # Group D 浅绿
]

## For Gumbel-max watermarks
# colors = ["tab:blue", "tab:orange",  "tab:red", "tab:brown", "tab:gray", "black",  "tab:purple",  "tab:pink",]
linestyles = [":", (0, (3, 1, 1, 1)),"-.", "--", "-"]

first = 700
name = f"results_data/K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-max{top_prob}"
with open(name+"-result.json", "r") as f:
    result_dict = json.load(f)
with open(name+"-null.json", "r") as f:
    null_dict = json.load(f)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,7))

for j, algo in enumerate(["ars", "opt-0005"]):
    orignal, unique = null_dict[algo]
    x = np.arange(1, len(orignal)+1)
    ax[0][0].plot(x, np.array(orignal), label="raw " + labelize(algo), linestyle=linestyles[j], color=colors[j])

for j, algo in enumerate(["ars", "opt-0005", "rep-opt-0005"]):
    if "rep" not in algo:
        orignal, unique = null_dict[algo]
    else:
        unique = null_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[0][0].plot(x, unique, label="deduped "+labelize(algo), linestyle=linestyles[2+j],color=colors[2+j])

ax[0][0].set_title(r"$H_0$")
ax[0][0].axhline(y=0.01, color="black", linestyle="dotted")
ax[0][0].set_ylabel(r"Type I error")
ax[0][0].set_xlabel(r"Number of tokens (or minimal units)")
# ax[0][0].set_yscale('log')

for j, algo in enumerate(["ars", "opt-0005",]):
    orignal, unique = result_dict[algo]
    x = np.arange(1, len(orignal)+1)
    ax[0][1].plot(x[100:], 1-np.array(orignal)[100:], label="raw "+labelize(algo), linestyle=linestyles[j],color=colors[j])

for j, algo in enumerate(["ars", "opt-0005", "rep-opt-0005"]):
    if "rep" not in algo:
        orignal, unique = result_dict[algo]
    else:
        unique = result_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[0][1].plot(x[100:], 1-np.array(unique)[100:], label="deduped "+labelize(algo), linestyle=linestyles[2+j],color=colors[2+j])

# for j, algo in enumerate(["ars", "opt-0005", "rep-opt-0005"]):
#     if "rep" not in algo:
#         orignal, unique = result_dict[algo]
#         x = np.arange(1, len(orignal)+1)
#         ax[0][1].plot(x[100:], 1-np.array(orignal)[100:], label=labelize(algo)+" (raw)", linestyle=":",color=colors[j%len(colors)])
#     else:
#         unique = result_dict[algo]
#     x = np.arange(1, len(unique)+1)
#     ax[0][1].plot(x[100:], 1-np.array(unique)[100:], label=labelize(algo) + " (deduped)", linestyle="-",color=colors[j%len(colors)])

ax[0][1].set_title(rf"$H_1, \Delta \sim U(0.001, {Delta})$")
ax[0][1].set_ylabel(r"Type II error")
ax[0][1].set_xlabel(r"Number of tokens (or minimal units)")
# ax[0][1].set_ylim(top=0.2)
if use_log:
    ax[0][1].set_yscale('log')
# ax[0][1].legend()
ax[0][1].legend(
    loc="center left",      # place legend to the left/right of axes
    bbox_to_anchor=(1, 0.5) # x=1 means right edge of axes, y=0.5 centers vertically
)


name = f"results_data/K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-inv{top_prob}"
with open(name+"-result.json", "r") as f:
    result_dict = json.load(f)
with open(name+"-null.json", "r") as f:
    null_dict = json.load(f)

for j, algo in enumerate(["dif", "dif-opt-001"]):
    k = 5 if j == 0 else 0
    orignal, unique = null_dict[algo]
    x = np.arange(1, len(orignal)+1)
    ax[1][0].plot(x[k:], np.array(orignal)[k:], label="raw " + labelize_inv(algo), linestyle=linestyles[j], color=colors[j])

for j, algo in enumerate(["dif", "dif-opt-001", "rep-dif-opt-001"]):
    k = 5 if j == 0 else 0
    if "rep" not in algo:
        orignal, unique = null_dict[algo]
    else:
        unique = null_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[1][0].plot(x[k:], np.array(unique)[k:], label="deduped "+labelize_inv(algo), linestyle=linestyles[2+j],color=colors[2+j])

# for j, algo in enumerate(["dif", "dif-opt-001", "rep-dif-opt-001"]):
#     if j == 0:
#         k = 5
#     else:
#         k = 0
#     if "rep" not in algo:
#         orignal, unique = null_dict[algo]
#         x = np.arange(1, len(orignal)+1)
#         ax[1][0].plot(x[k:],np.array(orignal)[k:], label=labelize_inv(algo)+" (raw)", linestyle=":",color=colors[j%len(colors)])
#     else:
#         unique = null_dict[algo]

#     x = np.arange(1, len(unique)+1)
#     ax[1][0].plot(x[k:], np.array(unique)[k:], label=labelize_inv(algo) + " (deduped)", linestyle="-",color=colors[j%len(colors)])


ax[1][0].set_title(r"$H_0$")
ax[1][0].axhline(y=0.01, color="black", linestyle="dotted")
ax[1][0].set_ylabel(r"Type I error")
ax[1][0].set_xlabel(r"Number of tokens (or minimal units)")
# ax[1][0].set_yscale('log')

for j, algo in enumerate(["dif", "dif-opt-001"]):
    k = 100
    orignal, unique = result_dict[algo]
    x = np.arange(1, len(orignal)+1)
    ax[1][1].plot(x[k:], 1-np.array(orignal)[k:], label="raw " + labelize_inv(algo), linestyle=linestyles[j], color=colors[j])

for j, algo in enumerate(["dif", "dif-opt-001", "rep-dif-opt-001"]):
    k = 100
    if "rep" not in algo:
        orignal, unique = result_dict[algo]
    else:
        unique = result_dict[algo]
    x = np.arange(1, len(unique)+1)
    ax[1][1].plot(x[k:], 1-np.array(unique)[k:], label="deduped "+labelize_inv(algo), linestyle=linestyles[2+j],color=colors[2+j])


# for j, algo in enumerate(["dif", "dif-opt-001", "rep-dif-opt-001"]):
#     if "rep" not in algo:
#         orignal, unique = result_dict[algo]
#         x = np.arange(1, len(orignal)+1)
#         ax[1][1].plot(x[100:], 1-np.array(orignal)[100:], label=labelize_inv(algo)+ " (raw)", linestyle=":",color=colors[j%len(colors)])
#     else:
#         unique = result_dict[algo]
#     x = np.arange(1, len(unique)+1)
#     ax[1][1].plot(x[100:], 1-np.array(unique)[100:], label=labelize_inv(algo)+ " (deduped)", linestyle="-",color=colors[j%len(colors)])

ax[1][1].set_title(rf"$H_1, \Delta \sim U(0.001, {Delta})$")
ax[1][1].set_ylabel(r"Type II error")
ax[1][1].set_xlabel(r"Number of tokens (or minimal units)")
if use_log:
    ax[1][1].set_yscale('log')
# ax[1][1].legend()
ax[1][1].legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

# plt.tight_layout()
plt.tight_layout(rect=[0, 0, 0.99, 1])

plt.savefig(f'K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-all{top_prob}.pdf', dpi=300)
