import numpy as np

# ----------------------------- Core single-y score (mk=1) -----------------------------

def compute_h_inv_per_y(Y, delta_vm, clip=None, eps=1e-6):
    """
    Elementwise optimal score for the inverse model using the m_k=1 case.

    h(y) = log( f_Δ(y) / f0(y) ), where
      f0(y) = 2(1 - y) * 1_{0 ≤ y ≤ 1}
      fΔ(y) = [ 2/(1-Δ) - 2y/(1-Δ)^2 ] * 1_{0 < y < 1-Δ}

    Vectorized over Y. Since Python 3, you can use Unicode letters (including Greek Δ) in variable names.
    """
    Y = np.asarray(Y, dtype=float)
    Δ = float(delta_vm)
    if not (0.0 < Δ < 1.0):
        raise ValueError("delta_vm must be in (0,1).")

    one_minus = 1.0 - Δ
    f0 = np.where((Y >= 0.0) & (Y <= 1.0), 2.0 * (1.0 - Y), 0.0)
    f1_core = 2.0 / one_minus - 2.0 * Y / (one_minus ** 2)
    f1 = np.where((Y > 0.0) & (Y < one_minus), f1_core, 0.0)

    h = np.log(np.maximum(f1, eps)) - np.log(np.maximum(f0, eps))
    if clip is not None:
        lo, hi = clip
        h = np.clip(h, lo, hi)
    return h, {"f0": f0, "f1": f1}

# ----------------------------- Helpers for block score -----------------------------

def _f0_density(y):
    """
    Exact f0(y) via piecewise integration:
      f0(y) = ∫_0^1 2^{|I1(u)|} * 1_{I2(u)=∅} du,
    with I1/I2 as in the theorem. Works for any length m>=1. 
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return 1.0
    if np.any((y < 0) | (y > 1)):
        return 0.0

    bps = {0.0, 0.5, 1.0}
    for yi in y:
        bps.add(float(yi))
        bps.add(float(1.0 - yi))
    bps = sorted(bps)

    total = 0.0
    for a, b in zip(bps[:-1], bps[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        # I2(mid) empty?
        if np.any((y >= 0.5) & (mid >= (1.0 - y)) & (mid <= y)):
            continue
        # |I1(mid)|
        N = int(np.sum((mid > y) & (mid < (1.0 - y))))
        total += (b - a) * (2.0 ** N)
    return float(total)

def _I_m_normalization(delta: float, m: int, quad_n: int = 128) -> float:
    """
    I_m(Δ) per Lemma:
      - for 0 ≤ Δ ≤ 1/2:  1 - (2m)/(m+1) * Δ
      - for 1/2 < Δ < 1:  ((1-Δ)/Δ)^m * (1 - 2m(1-Δ)/(m+1))
    (The formula also gives I_1(Δ)=1-Δ.)
    """
    Δ = float(delta)
    if not (0.0 < Δ < 1.0):
        raise ValueError("delta_vm must be in (0,1).")
    if Δ <= 0.5:
        val = 1.0 - (2.0 * m) / (m + 1.0) * Δ
    else:
        val = ((1.0 - Δ) / Δ) ** m * (1.0 - (2.0 * m * (1.0 - Δ)) / (m + 1.0))
    return float(max(val, 0.0))

class _SigmaCache:
    """Cache sign matrices σ ∈ {-1,1}^{2^m × m} and normalization constants."""
    def __init__(self):
        self.sigma = {}   # m -> (2^m, m) array
        self.Im = {}      # (m, Δ) -> float

    def get_sigma(self, m):
        if m in self.sigma:
            return self.sigma[m]
        # Build σ rows by counting in binary
        rows = 1 << m
        s = np.empty((rows, m), dtype=float)
        for j in range(m):
            # pattern length 2^j: -1 repeated, then +1 repeated, tiled
            block = 1 << j
            pattern = np.concatenate([np.full(block, -1.0), np.full(block, 1.0)])
            s[:, j] = np.tile(pattern, rows // (2 * block))
        self.sigma[m] = s
        return s

    def get_Im(self, m, Δ):
        key = (m, float(Δ))
        if key in self.Im:
            return self.Im[key]
        val = _I_m_normalization(Δ, m)
        self.Im[key] = val
        return val

_SIG_CACHE = _SigmaCache()

# --- Alternative density f_Δ(y): vectorized sum over ALL σ ∈ {−1,1}^m ---
def _f_alt_equal_delta_vectorized(y: np.ndarray, delta: float) -> float:
    """
    f_Δ(y) = (1 / I_m(Δ)) * Σ_{σ∈{−1,1}^m} (B_σ^Δ(y) − A_σ^Δ(y))_+
    with
      L_σ(y) = max_i { -σ_i y_i },  U_σ(y) = min_i { 1 − σ_i y_i },
      Y_σ^+(y) = max( { Δ y_i / (1−Δ) : σ_i=+1 } ∪ {0} ),
      Y_σ^−(y) = max( { Δ y_i / (1−Δ) : σ_i=−1 } ∪ {0} ),
      A_σ^Δ = max{L_σ, Y_σ^+},  B_σ^Δ = min{U_σ, 1 − Y_σ^− }.
    """
    y = np.asarray(y, dtype=float)
    m = y.shape[0]
    if m == 0:
        return 1.0
    if np.any((y < 0.0) | (y > 1.0)):
        return 0.0

    Δ = float(delta)
    σ = _SIG_CACHE.get_sigma(m)     # shape (2^m, m): ALL sign vectors
    Im = _SIG_CACHE.get_Im(m, Δ)
    if Im <= 0.0:
        return 0.0

    coef = Δ / (1.0 - Δ)

    # Per-σ L and U
    L = np.max(-σ * y, axis=1)          # (2^m,)
    U = np.min(1.0 - σ * y, axis=1)     # (2^m,)

    # Masks for σ_i = +1 / −1
    mask_pos = (σ > 0)
    mask_neg = ~mask_pos

    # Per-σ Y_σ^+ and Y_σ^- (max over selected coordinates; default 0 if set is empty)
    with np.errstate(invalid="ignore"):
        pos_vals = np.where(mask_pos, y, -np.inf)  # (2^m, m)
        neg_vals = np.where(mask_neg, y, -np.inf)

    Yp = np.where(
        np.any(mask_pos, axis=1),
        coef * np.max(pos_vals, axis=1),
        0.0,
    )
    Ym = np.where(
        np.any(mask_neg, axis=1),
        coef * np.max(neg_vals, axis=1),
        0.0,
    )
    # A_σ^Δ and B_σ^Δ, then lengths (B−A)_+
    A = np.maximum(L, Yp)
    B = np.minimum(U, 1.0 - Ym)
    lengths = np.maximum(B - A, 0.0)    # (2^m,)

    # Sum over ALL σ and normalize
    return float(np.sum(lengths) / Im)

def _block_score(y, delta_vm, clip=None, eps=1e-300):
    """h_V^{inv}(y) = log(f_Δ(y)/f0(y)) for a single y-vector (block)."""
    f0 = _f0_density(y)
    f1 = _f_alt_equal_delta_vectorized(y, delta_vm)
    h = np.log(max(f1, eps)) - np.log(max(f0, eps))
    if clip is not None:
        lo, hi = clip
        h = float(np.clip(h, lo, hi))
    return h, f0, f1

# ----------------------------- Public API -----------------------------

import numpy as np

import numpy as np

def compute_scores_with_prefixes(Y, group, delta_vm, clip=None):
    """
    Per-entry singleton scores (m_k=1) and group-based block scores.

    Inputs
    ------
    Y      : shape (n,) or (R, n)
    group  : shape (n,)  (shared across rows when Y is 2-D)
    delta_vm : float in (0,1)
    clip   : optional (lo, hi) to clip log-scores

    Returns (1-D)
    ------------
    {
      "per_y_score":                        (n,),
      "per_y_cumsum":                       (n,),
      "group_block_scores":                 (G,),      # per group (first-appearance order)
      "group_block_scores_cumsum":          (G,),      # cumsum over groups
      "group_block_scores_cumsum_by_pos":   (n,),      # step fn over positions
      "unique_groups_in_order":             list of length G
    }

    Returns (2-D)
    ------------
    Same keys, with shapes:
      per_y_score/per_y_cumsum:             (R, n)
      group_block_scores:                   (R, G)
      group_block_scores_cumsum:            (R, G)
      group_block_scores_cumsum_by_pos:     (R, n)
    """
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(group)

    # ---------- 1-D ----------
    if Y.ndim == 1:
        n = Y.shape[0]
        if group.shape[0] != n:
            raise ValueError("Y and group must have the same length.")

        # (1) Per-entry (m_k=1) scores & cumsum (vectorized)
        per_y_score, _ = compute_h_inv_per_y(Y, delta_vm, clip=clip)
        per_y_cumsum = np.cumsum(per_y_score)

        # (2) Index lists per group + first appearances
        idxs_by_group = {}
        first_idx = {}
        for i, g in enumerate(group):
            idxs_by_group.setdefault(g, []).append(i)
            if g not in first_idx:
                first_idx[g] = i
        groups_sorted = sorted(first_idx.keys(), key=lambda g: first_idx[g])
        G = len(groups_sorted)

        # Split into singleton vs multi-element groups
        singleton_groups = [g for g in groups_sorted if len(idxs_by_group[g]) == 1]
        multi_groups     = [g for g in groups_sorted if len(idxs_by_group[g]) > 1]

        # (3) Per-group block scores (fast path for singletons)
        group_block_scores = np.empty(G, dtype=float)

        # Singletons: h(y) from mk=1 in one vectorized call
        if singleton_groups:
            single_pos = np.array([idxs_by_group[g][0] for g in singleton_groups], dtype=int)
            single_scores, _ = compute_h_inv_per_y(Y[single_pos], delta_vm, clip=clip)  # (S,)
            # place back into group order
            score_map = {g: s for g, s in zip(singleton_groups, single_scores)}
        else:
            score_map = {}

        # Fill group_block_scores: mk=1 for singletons, _block_score for multis
        for j, g in enumerate(groups_sorted):
            if g in score_map:
                group_block_scores[j] = score_map[g]
            else:
                y_g = Y[np.array(idxs_by_group[g], dtype=int)]
                h_g, _, _ = _block_score(y_g, delta_vm, clip=clip)
                group_block_scores[j] = h_g

        # (4) Prefix over groups (cumsum in group order)
        group_block_scores_cumsum = np.cumsum(group_block_scores)  # (G,)

        # (5) Map to positions as a step function (increments at first appearances)
        increments_by_pos = np.zeros(n, dtype=float)
        for j, g in enumerate(groups_sorted):
            i0 = first_idx[g]
            increments_by_pos[i0] = group_block_scores[j]
        group_block_scores_cumsum_by_pos = np.cumsum(increments_by_pos)  # (n,)

        return {
            "per_y_score": per_y_score,                                    # (n,)
            "per_y_cumsum": per_y_cumsum,                                  # (n,)
            "group_block_scores": group_block_scores,                      # (G,)
            "group_block_scores_cumsum": group_block_scores_cumsum,        # (G,)
            "group_block_scores_cumsum_by_pos": group_block_scores_cumsum_by_pos,  # (n,)
            "unique_groups_in_order": groups_sorted,                       # list length G
        }

    # ---------- 2-D ----------
    elif Y.ndim == 2:
        R, n = Y.shape
        if group.shape[0] != n:
            raise ValueError("group must have length n == Y.shape[1].")

        # (1) Per-entry (m_k=1) scores & row-wise cumsum (vectorized)
        per_y_score, _ = compute_h_inv_per_y(Y, delta_vm, clip=clip)       # (R, n)
        per_y_cumsum = np.cumsum(per_y_score, axis=1)                      # (R, n)

        # (2) Index lists per group + first appearances (shared across rows)
        idxs_by_group = {}
        first_idx = {}
        for i, g in enumerate(group):
            idxs_by_group.setdefault(g, []).append(i)
            if g not in first_idx:
                first_idx[g] = i
        groups_sorted = sorted(first_idx.keys(), key=lambda g: first_idx[g])
        G = len(groups_sorted)

        singleton_groups = [g for g in groups_sorted if len(idxs_by_group[g]) == 1]
        multi_groups     = [g for g in groups_sorted if len(idxs_by_group[g]) > 1]

        # (3) Per-group block scores per row (fast path for singletons)
        group_block_scores = np.empty((R, G), dtype=float)

        # Singletons: vectorized mk=1 over rows
        if singleton_groups:
            single_cols = np.array([idxs_by_group[g][0] for g in singleton_groups], dtype=int)
            single_scores, _ = compute_h_inv_per_y(Y[:, single_cols], delta_vm, clip=clip)  # (R, S)
            # map singleton group -> column index inside single_cols
            col_map = {g: c for c, g in enumerate(singleton_groups)}
        else:
            col_map = {}

        for j, g in enumerate(groups_sorted):
            cols = idxs_by_group[g]
            if g in col_map:
                # singleton, pull from vectorized mk=1 result
                group_block_scores[:, j] = single_scores[:, col_map[g]]
            else:
                cols = np.array(cols, dtype=int)
                for r in range(R):
                    y_g = Y[r, cols]
                    h_g, _, _ = _block_score(y_g, delta_vm, clip=clip)
                    group_block_scores[r, j] = h_g

        # (4) Prefix over groups (row-wise cumsum)
        group_block_scores_cumsum = np.cumsum(group_block_scores, axis=1)  # (R, G)

        # (5) Map to positions: put increments at first appearances, cumsum along axis=1
        increments_by_pos = np.zeros((R, n), dtype=float)
        for j, g in enumerate(groups_sorted):
            i0 = first_idx[g]
            increments_by_pos[:, i0] = group_block_scores[:, j]
        group_block_scores_cumsum_by_pos = np.cumsum(increments_by_pos, axis=1)  # (R, n)

        return {
            "per_y_score": per_y_score,                                      # (R, n)
            "per_y_cumsum": per_y_cumsum,                                    # (R, n)
            "group_block_scores": group_block_scores,                        # (R, G)
            "group_block_scores_cumsum": group_block_scores_cumsum,          # (R, G)
            "group_block_scores_cumsum_by_pos": group_block_scores_cumsum_by_pos,  # (R, n)
            "unique_groups_in_order": groups_sorted,                         # list length G
        }

    else:
        raise ValueError("Y must be 1-D or 2-D.")

# --------------------------------- Example ---------------------------------
if __name__ == "__main__":
    # Example: Y with repeated groups; only the first time a new group appears
    # does the block prefix actually grow.
    Y = np.array([0.20, 0.35, 0.10, 0.20, 0.35, 0.12, 0.08, 0.09])
    group = np.array([  0,    1,    2,    0,    1,    3,    4,    5])
    delta_vm = 0.1

    out = compute_scores_with_prefixes(Y, group, delta_vm)
    print("per_y_score:", out["per_y_score"])
    print("per_y_cumsum:", out["per_y_cumsum"])
    # print("group_block_scores_cumsum_by_pos:", out["group_block_scores_cumsum_by_pos"])
    print("unique order:", out["unique_groups_in_order"])
    print("group_block_scores_cumsum:", out["group_block_scores_cumsum"])

    # Build a 2-D Y: shape (R=3, n=8)
    # For each row: make sure Y[:, idx] are equal for duplicate groups:
    #   group 0 at idx {0,3}, group 1 at {1,4}, group 4 at {6,7}
    Y2 = np.array([
        [0.20, 0.35, 0.10, 0.20, 0.35, 0.12, 0.08, 0.08],  # row 0 (your 1-D example)
        [0.25, 0.30, 0.15, 0.25, 0.30, 0.18, 0.05, 0.05],  # row 1
        [0.10, 0.40, 0.12, 0.10, 0.40, 0.22, 0.09, 0.09],  # row 2
    ], dtype=float)

    out = compute_scores_with_prefixes(Y2, group, delta_vm)

    print("per_y_score shape:", out["per_y_score"].shape)                     # (3, 8)
    print("per_y_cumsum shape:", out["per_y_cumsum"].shape)                   # (3, 8)
    # print("group_block_scores_cumsum_by_pos shape:", out["group_block_scores_cumsum_by_pos"].shape)  # (3, 8)
    print("group_block_scores_cumsum shape:", out["group_block_scores_cumsum"].shape)  # (3, G)

    # Peek at results
    print("\nRow 0 per_y_cumsum:", out["per_y_cumsum"][0])
    print("Row 1 per_y_cumsum:", out["per_y_cumsum"][1])
    print("Row 2 per_y_cumsum:", out["per_y_cumsum"][2])

    # print("\nRow 0 group_block_scores_cumsum_by_pos:", out["group_block_scores_cumsum_by_pos"][0])
    # print("Row 1 group_block_scores_cumsum_by_pos:", out["group_block_scores_cumsum_by_pos"][1])
    # print("Row 2 group_block_scores_cumsum_by_pos:", out["group_block_scores_cumsum_by_pos"][2])

    print("\nUnique order (shared across rows):", out["unique_groups_in_order"])
    print("Row 0 group_block_scores_cumsum:", out["group_block_scores_cumsum"][0])
    print("Row 1 group_block_scores_cumsum:", out["group_block_scores_cumsum"][1])
    print("Row 2 group_block_scores_cumsum:", out["group_block_scores_cumsum"][2])
