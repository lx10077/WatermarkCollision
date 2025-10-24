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
    delta_vm = np.asarray(delta_vm, dtype=float)
    if not np.all((0.0 < delta_vm) & (delta_vm < 1.0)):
        raise ValueError("All delta_vm values must be in (0,1).")

    one_minus = 1.0 - delta_vm

    f0 = np.where((Y >= 0.0) & (Y <= 1.0), 2.0 * (1.0 - Y), 0.0)  # (n,)
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


def compute_scores_with_prefixes(Y, group, PS, clip=None):
    """
    Vectorized version supporting 1D or 2D Y.

    Parameters
    ----------
    Y : array-like, shape (n,) or (R, n)
    group : array-like, shape (n,)
    PS : array-like, shape (n,)
    clip : tuple (lo, hi) or None

    Returns
    -------
    dict with same keys as before; shapes:
      - per_y_score, per_y_cumsum: (R, n)
      - group_block_scores, group_block_scores_cumsum: (R, G)
      - group_block_scores_cumsum_by_pos: (R, n)
    """
    Y = np.asarray(Y, dtype=float)
    PS = np.asarray(PS, dtype=float)
    group = np.asarray(group)

    # unify shape handling
    if Y.ndim == 1:
        Y = Y[None, :]  # promote to (1, n)
    R, n = Y.shape
    if PS.shape[0] != n or group.shape[0] != n:
        raise ValueError("group and PS must have same length as Y.shape[1]")

    # ---------- (1) group indexing ----------
    idxs_by_group = {}
    first_idx = {}
    for i, g in enumerate(group):
        idxs_by_group.setdefault(g, []).append(i)
        if g not in first_idx:
            first_idx[g] = i
    groups_sorted = sorted(first_idx.keys(), key=lambda g: first_idx[g])
    G = len(groups_sorted)

    # ---------- (2) δ_g = 1 - max(PS[group]) ----------
    delta_map = {g: 1.0 - np.max(PS[idxs_by_group[g]]) for g in groups_sorted}
    deltas_per_y = np.array([delta_map[g] for g in group])  # (n,)

    # ---------- (3) per-entry scores ----------
    # 假设 compute_h_inv_per_y 支持 vectorized 调用: (R, n) × (n,)
    per_y_score = compute_h_inv_per_y(Y, deltas_per_y, clip=clip)[0]  # (R, n)
    per_y_cumsum = np.cumsum(per_y_score, axis=1)

    # ---------- (4) per-group block scores ----------
    group_block_scores = np.empty((R, G), dtype=float)
    for j, g in enumerate(groups_sorted):
        cols = np.array(idxs_by_group[g], dtype=int)
        delta_g = delta_map[g]
        if len(cols) == 1:
            h_g, _ = compute_h_inv_per_y(Y[:, cols], delta_g, clip=clip)
            group_block_scores[:, j] = h_g.squeeze()
        else:
            # vectorized _block_score across rows
            all_scores = [_block_score(Y[r, cols], delta_g, clip=clip)[0] for r in range(R)]
            group_block_scores[:, j] = np.array(all_scores)

    group_block_scores_cumsum = np.cumsum(group_block_scores, axis=1)

    # ---------- (5) prefix map by position ----------
    increments_by_pos = np.zeros((R, n), dtype=float)
    for j, g in enumerate(groups_sorted):
        i0 = first_idx[g]
        increments_by_pos[:, i0] = group_block_scores[:, j]
    group_block_scores_cumsum_by_pos = np.cumsum(increments_by_pos, axis=1)

    return {
        "per_y_score": per_y_score,
        "per_y_cumsum": per_y_cumsum,
        "group_block_scores": group_block_scores,
        "group_block_scores_cumsum": group_block_scores_cumsum,
        "group_block_scores_cumsum_by_pos": group_block_scores_cumsum_by_pos,
        "unique_groups_in_order": groups_sorted,
        "group_deltas": delta_map,
    }

# --------------------------------- Example ---------------------------------
if __name__ == "__main__":

    # === Test 1: Distinct Singletons ===
    print("\n=== Test 1: singleton_groups ===")
    Y = np.array([0.1, 0.2, 0.3, 0.4])
    group = np.array([0, 1, 2, 3])
    PS = np.array([0.5, 0.6, 0.7, 0.8])
    out = compute_scores_with_prefixes(Y, group, PS)
    print("group_deltas:", out["group_deltas"])

    # === Test 2: Two Shared Groups ===
    print("\n=== Test 2: two_shared_groups ===")
    Y = np.array([0.2, 0.35, 0.1, 0.2, 0.35, 0.12])
    group = np.array([0, 1, 2, 0, 1, 3])
    PS = np.array([0.8, 0.7, 0.4, 0.9, 0.5, 0.6])
    out = compute_scores_with_prefixes(Y, group, PS)
    print("group_deltas:", out["group_deltas"])

    # === Test 3: Overlapping Groups ===
    print("\n=== Test 3: overlapping_groups ===")
    Y = np.array([0.15, 0.1, 0.25, 0.3, 0.12, 0.22, 0.18, 0.27])
    group = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    PS = np.array([0.9, 0.85, 0.4, 0.6, 0.3, 0.2, 0.25, 0.5])
    out = compute_scores_with_prefixes(Y, group, PS)
    print("group_deltas:", out["group_deltas"])

    # === Test 4: Mixed Group Sizes ===
    print("\n=== Test 4: mixed_group_sizes ===")
    Y = np.array([0.05, 0.25, 0.22, 0.4, 0.18, 0.3, 0.12, 0.07, 0.09])
    group = np.array([0, 1, 1, 1, 2, 2, 3, 4, 5])
    PS = np.array([0.6, 0.9, 0.7, 0.5, 0.4, 0.45, 0.2, 0.3, 0.25])
    out = compute_scores_with_prefixes(Y, group, PS)
    print("group_deltas:", out["group_deltas"])
