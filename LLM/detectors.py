import numpy as np
from scipy.stats import gamma, norm
from tqdm import tqdm
from score_inv_inhomo import compute_scores_with_prefixes

# ----------------------------------
# Help functions
# ----------------------------------

def rowwise_unique_topk_multi(A, *others, k=200):
    """Row-wise extract top-k unique elements with counts and auxiliary arrays."""
    A_rows, O_rows_list, C_rows = [], [[] for _ in others], []
    n, m = A.shape

    for i in range(n):
        seen = {}
        row_vals = []
        row_others = [[] for _ in others]
        row_counts = []

        for j in range(m):
            val = A[i, j]
            other_vals = [O[i, j] for O in others]

            if val not in seen:
                seen[val] = [other_vals[:], 1]  # Store [values from other matrices, count]
                row_vals.append(val)
                for kx, ov in enumerate(other_vals):
                    row_others[kx].append(ov)
                row_counts.append(1)
                if len(row_vals) == k:
                    break
            else:
                seen[val][1] += 1
                pos = row_vals.index(val)

                # Update maximum value for each other matrix
                for kx, ov in enumerate(other_vals):
                    if ov > seen[val][0][kx]:
                        seen[val][0][kx] = ov
                        row_others[kx][pos] = ov

                row_counts[pos] = seen[val][1]

        if len(row_vals) == k:
            A_rows.append(row_vals)
            for kx in range(len(others)):
                O_rows_list[kx].append(row_others[kx])
            C_rows.append(row_counts)

    A2 = np.array(A_rows, dtype=A.dtype)
    O2s = [np.array(rows, dtype=O.dtype) for rows, O in zip(O_rows_list, others)]
    C2 = np.array(C_rows, dtype=np.int64)

    return (A2, *O2s, C2)


def rough_upper_bound(generated_top_probs):
    """Compute rough quantized upper bounds for probability values."""
    endpoints = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
    indices = np.searchsorted(endpoints, generated_top_probs, side='right')
    indices = np.clip(indices, 0, len(endpoints) - 1)
    return endpoints[indices]


def pad_with_uniform(null_Ys, low=0.0, high=1.0, target_len=500):
    """Pad shorter rows with uniform random values."""
    d, n = null_Ys.shape
    if n >= target_len:
        return null_Ys[:, :target_len]
    pad_width = target_len - n
    pad = np.random.uniform(low, high, size=(d, pad_width))
    return np.hstack([null_Ys, pad])

# ----------------------------------
# Previous Detection Functions
# ----------------------------------

def LogDetector(Ys, alpha=0.05):
    """Classical log-based detector.

    Args:
        Ys: 2D array of test statistics.
        alpha: Significance level.
    """
    check_points = np.arange(1, 1 + Ys.shape[-1])
    h_log_qs = -gamma.ppf(alpha, a=check_points)
    log_Ys = np.log(np.array(Ys))
    cumsum_Ys = np.cumsum(log_Ys, axis=1)
    return check_points, (cumsum_Ys >= h_log_qs).mean(axis=0)


def ArsDetector(Ys, alpha=0.05):
    """ARS-based detector using gamma quantiles."""
    check_points = np.arange(1, 1 + Ys.shape[-1])
    h_ars_qs = gamma.ppf(1 - alpha, a=check_points)
    ars_Ys = -np.log(1 - np.array(Ys))
    cumsum_Ys = np.cumsum(ars_Ys, axis=1)
    return check_points, (cumsum_Ys >= h_ars_qs).mean(axis=0)


def f_opt(r, delta):
    inte = np.floor(1 / (1 - delta))
    rest = 1 - (1 - delta) * inte
    return np.log(inte * r ** (delta / (1 - delta)) + r ** (1 / rest - 1))


def OptGumDetector(Ys, delta0=0.1, alpha=0.01):
    """Optimal detector for Gumbel-max watermark."""
    Ys = np.array(Ys)
    h_opt_Ys = f_opt(Ys, delta0)

    def find_q(N=2500, delta0=delta0):
        Null_Ys = np.random.uniform(size=(N, Ys.shape[1]))
        Simu_Y = f_opt(Null_Ys, delta0)
        Simu_Y = np.cumsum(Simu_Y, axis=1)
        return np.quantile(Simu_Y, 1 - alpha, axis=0)

    q_lst = [find_q(2500) for _ in range(10)]
    h_help_qs = np.mean(np.array(q_lst), axis=0)
    cumsum_Ys = np.cumsum(h_opt_Ys, axis=1)
    return np.arange(1, 1 + Ys.shape[-1]), (cumsum_Ys >= h_help_qs).mean(axis=0)


def TrGofDetector(Ys, alpha=0.01, mask=True, eps=1e-10):
    """Truncated Goodness-of-Fit (TrGoF) test.

    Reference:
        Li et al. (2024) TrGoF: A Truncated Goodness-of-Fit Test for LLM Watermarking.
        arXiv: https://arxiv.org/abs/2411.13868
    """

    def compute_score(Ys_slice):
        ps = 1 - Ys_slice
        ps = np.sort(ps, axis=-1)
        m = ps.shape[-1]
        rk = np.arange(1, 1 + m) / m
        final = (rk - ps) ** 2 / (ps * (1 - ps) + eps) / 2
        if mask:
            valid = (ps >= 1e-3) * (rk >= ps)
            final *= valid
        return np.log(m * np.max(final, axis=-1) + 1e-10)

    def compute_quantile(m):
        qs = []
        for _ in range(10):
            raw_data = np.random.uniform(size=(2500, m))
            H0_scores = compute_score(raw_data)
            q = np.quantile(H0_scores, 1 - alpha)
            qs.append(q)
        return np.mean(qs)

    detection_curve = []
    xs = []
    for t in range(1, Ys.shape[1] + 1, 2):
        xs.append(t)
        Ys_trunc = Ys[:, :t]
        scores = compute_score(Ys_trunc)
        q = compute_quantile(t)
        detection_curve.append(np.mean(scores >= q))

    return np.array(xs), np.array(detection_curve)


# ----------------------------------
# Proposed Methods for Gumbel-max
# ----------------------------------

def RepeatWeightLogGumDetectorNew(Ys, Ps, alpha=0.01, delta=0.1):
    """Repeat-weighted log detector for Gumbel-max watermark."""

    def compute_score(Ys, largest_Pros):
        weight = (1 / (largest_Pros + 1e-6) - 1)
        another_weight = (1 / (1 - largest_Pros + 1e-6) - 1)
        return np.log(Ys ** weight + Ys ** another_weight + 1e-6)

    def compute_critical_values(largest_Pros):
        length = len(largest_Pros)
        all_qs = []
        for _ in range(10):
            null_U = np.random.uniform(size=(4000, length))
            shape = 1 / (largest_Pros + 1e-6) - 1
            another = 1 / (1 - largest_Pros + 1e-6) - 1
            qs = np.quantile(
                np.cumsum(np.log(null_U ** shape + null_U ** another + 1e-6), axis=1),
                1 - alpha,
                axis=0,
            )
            all_qs.append(qs)
        return np.mean(all_qs, axis=0)

    detection_results = []
    for i, sentence_Ys in tqdm(enumerate(Ys)):
        cleaned_Ys = Ys[i]
        largest_Pros = Ps[i]
        largest_Pros = np.maximum(np.minimum(largest_Pros, 1 - delta), delta)
        score_Ys = compute_score(cleaned_Ys, largest_Pros)
        cumsum_score_Ys = np.cumsum(score_Ys)
        critical_values = compute_critical_values(largest_Pros)
        detection_Ys = cumsum_score_Ys >= critical_values
        detection_results.append(detection_Ys)

    y = np.mean(detection_results, axis=0)
    return np.arange(1, 1 + len(y)), y


Delta1Star = np.array([0, 0, 0.29707273, 0.38798173, 0.42366049, 0.44200376, 0.45318023, 0.46069369, 0.46610054, 0.47020137])


def SwitchOptGumDetector(Ys, Ps, counts, alpha=0.01, delta=0.1, eps=1e-6):
    """Switching optimal detector for Gumbel-max watermark."""

    def compute_score1(Ys, largest_Pros):
        weight = (1 / (largest_Pros + 1e-6) - 1)
        another_weight = (1 / (1 - largest_Pros + 1e-6) - 1)
        return np.log(Ys ** weight + Ys ** another_weight + 1e-6)

    def compute_score2(Ys, largest_Pros):
        Ys_safe = np.clip(Ys, eps, 1.0)
        p_safe = np.clip(largest_Pros, eps, 1.0)
        weight = 1.0 / p_safe - 1
        return np.log(Ys_safe) * weight

    def compute_critical_values(largest_Pros, count):
        length = len(largest_Pros)
        all_qs = []
        for _ in range(10):
            null_U = np.random.uniform(size=(2500, length))
            score_Ys1 = compute_score1(null_U, largest_Pros)
            score_Ys2 = compute_score2(null_U, largest_Pros)
            score_Ys = np.where(
                (count >= 1)
                & (count <= len(Delta1Star))
                & ((1 - largest_Pros) >= np.take(Delta1Star, count - 1, mode="clip")),
                score_Ys1,
                score_Ys2,
            )
            qs = np.quantile(np.cumsum(score_Ys, axis=1), 1 - alpha, axis=0)
            all_qs.append(qs)
        return np.mean(all_qs, axis=0)

    detection_results = []
    for i, sentence_Ys in tqdm(enumerate(Ys)):
        cleaned_Ys, largest_Pros, count = Ys[i], Ps[i], counts[i]
        largest_Pros = np.maximum(np.minimum(largest_Pros, 1 - delta), delta)
        score_Ys1 = compute_score1(cleaned_Ys, largest_Pros)
        score_Ys2 = compute_score2(cleaned_Ys, largest_Pros)
        score_Ys = np.where(
            (count >= 1)
            & (count <= len(Delta1Star))
            & ((1 - largest_Pros) >= np.take(Delta1Star, count - 1, mode="clip")),
            score_Ys1,
            score_Ys2,
        )
        cumsum_score_Ys = np.cumsum(score_Ys)
        critical_values = compute_critical_values(largest_Pros, count)
        detection_Ys = cumsum_score_Ys >= critical_values
        detection_results.append(detection_Ys)

    return np.arange(1, 1 + Ys.shape[-1]), np.mean(detection_results, axis=0)


# ----------------------------------
# Proposed Methods for Inverse Model
# ----------------------------------

def f_inv(Y, delta, eps=1e-6):
    """Helper transformation for inverse model detection."""
    return np.log(np.maximum(1 + Y / (1 - delta), eps) / np.maximum(1 + Y, 0) / (1 - delta))


def OptInvDetector(Ys, vocab_size, delta0=0.1, alpha=0.01):
    """Optimal inverse detector based on vocabulary size."""
    Ys = -np.abs(Ys)
    h_abs_Ys = f_inv(Ys, delta0)

    def find_q(N=1000):
        Null_Ys_U = np.random.uniform(size=(N, Ys.shape[1]))
        Null_Ys_pi_s = np.random.randint(low=0, high=vocab_size, size=(N, Ys.shape[1]))
        Null_etas = np.array(Null_Ys_pi_s) / (vocab_size - 1)
        null_final_Y = -np.abs(Null_Ys_U - Null_etas)
        null_final_Y = f_inv(null_final_Y, delta0)
        null_cumsum_Ys = np.cumsum(null_final_Y, axis=1)
        return np.quantile(null_cumsum_Ys, 1 - alpha, axis=0)

    h_help_qs = find_q(2500)
    cumsum_Ys = np.cumsum(h_abs_Ys, axis=1)
    return np.arange(1, 1 + Ys.shape[-1]), (cumsum_Ys >= h_help_qs).mean(axis=0)


def compute_general_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t * mu + q * np.sqrt(t * var))
    return np.array(qs)


def IdInvDetector(Ds, alpha=0.01):
    """Identity-based inverse detector."""
    Ds = -np.abs(Ds)
    check_points = np.arange(1, 1 + Ds.shape[-1])
    mu_dif = -1 / 3
    var_dif = 1 / 6 - 1 / 9
    h_id_dif_qs = compute_general_q(1 - alpha, mu_dif, var_dif, check_points)
    cumsum_Ds = np.cumsum(Ds, axis=1)
    results = cumsum_Ds >= h_id_dif_qs
    return np.arange(1, 1 + Ds.shape[-1]), np.mean(results, axis=0)


def group_by_ds(Ys, Ds, tol=1e-6):
    """Group elements in Ds by rounding for tolerance comparison."""
    Ys = np.asarray(Ys)
    Ds = np.asarray(Ds)
    if Ys.shape != Ds.shape:
        raise ValueError("Ys and Ds must have the same shape")

    if Ds.ndim == 1:
        rounded = np.round(Ds / tol).astype(int)
        _, inverse_indices = np.unique(rounded, return_inverse=True)
        return inverse_indices

    elif Ds.ndim == 2:
        R, n = Ds.shape
        inverse_indices = np.empty_like(Ds, dtype=int)
        for r in range(R):
            rounded = np.round(Ds[r] / tol).astype(int)
            _, inverse_indices[r] = np.unique(rounded, return_inverse=True)
        return inverse_indices

    else:
        raise ValueError("Ys and Ds must be 1-D or 2-D arrays.")


def RepeatOptInvDetector(Ys, Ds, Ps, vocab_size, delta=0.01, alpha=0.01, eps=1e-6, min_count=220):
    """Repeated optimal inverse detector with grouping and prefix scoring."""
    Ys = np.abs(Ys)
    detection_results = []

    for i, sentence_Ys in tqdm(enumerate(Ys)):
        if np.unique(Ds[i]).size < min_count:
            continue

        group = group_by_ds(sentence_Ys, Ds[i], tol=eps)
        largest_Pros = Ps[i]
        largest_Pros = np.maximum(np.minimum(largest_Pros, 1 - delta), delta)

        result_dict = compute_scores_with_prefixes(sentence_Ys, group, largest_Pros)
        cumsum_score_Ys = result_dict["group_block_scores_cumsum"].squeeze()

        def find_q(N, group, largest_Pros):
            Null_Ys_U = np.random.uniform(size=(N, Ys.shape[1]))
            Null_Ys_pi_s = np.random.randint(low=0, high=vocab_size, size=(N, Ys.shape[1]))
            Null_etas = np.array(Null_Ys_pi_s) / (vocab_size - 1)
            null_final_Y = np.abs(Null_Ys_U - Null_etas)
            null_cumsum_Ys = compute_scores_with_prefixes(
                null_final_Y, group, largest_Pros
            )["group_block_scores_cumsum"]
            return np.quantile(null_cumsum_Ys, 1 - alpha, axis=0)

        critical_values = np.mean(
            [find_q(4000, group, largest_Pros) for _ in range(10)], axis=0
        )
        detection_Ys = cumsum_score_Ys >= critical_values
        detection_results.append(detection_Ys[:min_count])

    return np.arange(1, 1 + min_count), np.mean(detection_results, axis=0)


if __name__ == "__main__":
    Ys = np.array([[0.2, 0.3, 0.12, 0.1, 0.1, 0.4], [0.5, 0.1, 0.31, 0.1, 0.2, 0.3]])
    Ds = np.array(
        [
            [1.0, 1.0000002, 2.0, 2.0, 3.0, 4.0],
            [1.0, 1.0, 1.0, 2.0, 2.0, 6.0],
        ]
    )
    Ps = np.array(
        [
            [0.9, 0.8, 0.7, 0.9, 0.5, 0.4],
            [0.6, 0.5, 0.8, 0.9, 0.3, 0.2],
        ]
    )
    vocab_size = 50
    print(RepeatOptInvDetector(Ys, Ds, Ps, vocab_size, alpha=0.05, min_count=3))