import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Sequence
import random
from IPython import embed


# ---------- plug your own probability generator here ----------
def Zipf(a=1., b=0.01, support_size=5):
    support_Ps = np.arange(1, 1+support_size)
    support_Ps = (support_Ps + b)**(-a)
    support_Ps /= support_Ps.sum()
    return support_Ps

def dominate_Ps(Delta, K):
    if 1-Delta <= 1/K:
        return np.ones(K)/K
    a = np.random.uniform(0.95, 1.5)
    b = np.random.uniform(0.01,0.1)
    Head_Ps = Zipf(a=a, b=b, support_size=K-1)
    b = (1 - Delta)/Head_Ps.max()
    Ps = np.ones(K)
    Ps[0] = max(1-Delta, 1/K)
    Ps[1:] = Head_Ps * Delta
    assert Ps.max() <= 1-Delta+1e-5 and Ps.max() >= 1-Delta-1e-5 and np.abs(np.sum(Ps)- 1)<= 1e-3
    random.shuffle(Ps)
    return Ps

# ---------- local uniform generator ----------
def generate_uniform_local(inputs: List[int], c: int, key: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return two arrays:
      - xi: shape (1,), uniform(0,1)
      - pi: shape (K,), a permutation of {1, 2, ..., K}
    Reproducible given the last (c-1) tokens and key.
    """
    assert len(inputs) >= c
    tail = inputs[-(c-1):] if c > 1 else []
    # Seed Python RNG from (tail, key)
    random.seed(tuple(tail + [key]))
    # Use Python RNG to seed NumPy RNG (so both are tied to the same context)
    np.random.seed(random.randrange(2**32 - 1))

    xi = np.random.uniform(size=1)         # array (1,)
    pi = np.random.permutation(K)         # array (K,), values 1..K
    return xi[0], pi

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def find_next_token(xi, probs, pi):
    inv_pi = inv(pi)
    inv_probs = probs[inv_pi]
    i = 0
    s = 0
    while s <= xi:
        s += inv_probs[i]
        i += 1
    return inv_pi[i-1]

# ---------- the generator ----------
class Inv_repeated_token_generator:
    def __init__(
        self,
        vocab_size: int,
        max_window: int,
        Delta: float,
        c: int,
        key: int,
        p_types: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    ):
        assert vocab_size >= 1
        self.K = vocab_size
        self.max_window = max_window
        self.Delta = Delta
        self.c = c
        self.key = key

        # Store raw weights and a normalized copy for the cap logic
        self.p_types_raw = tuple(max(0.0, float(w)) for w in p_types)
        s = sum(self.p_types_raw)
        if s <= 0:
            # fallback: always force World-1
            self._target_fracs = (1.0, 0.0, 0.0)
        else:
            self._target_fracs = tuple(w / s for w in self.p_types_raw)

        self.tokens: List[int] = []
        self.Us: List[int] = []
        self.etas: List[float] = []
        self.difs: List[float] = []
        self.probs: List[float] = []
        self.saved_intervals: List[Dict] = []
        self.prompt_len: int = 0

    # -------- helpers --------
    # Optional: keep this for compatibility; uses raw weights (sum not required to be 1)
    def _sample_world(self) -> int:
        return random.choices([1, 2, 3], weights=self.p_types_raw, k=1)[0]

    def _choose_world_capped(self, counts: List[int], rounds_done: int) -> int:
        """
        Choose a world ensuring cumulative fractions never exceed normalized targets.
        If a world is unavailable or capped, it won't be chosen. If none allowed, force W1.
        """
        t = rounds_done  # completed steps
        allowed = []
        for w in (1, 2, 3):
            # availability
            if w == 2 and not self._can_copy_new_interval():
                continue
            if w == 3 and not self._can_copy_existing_interval():
                continue
            idx = w - 1
            target = self._target_fracs[idx]
            # cap: (counts[idx] + 1) / (t + 1) <= target
            if counts[idx] + 1 <= target * (t + 1) + 1e-12:
                allowed.append(w)

        if not allowed:
            return 1  # ensure progress

        # weight allowed worlds by *normalized* target fractions
        weights = [self._target_fracs[w - 1] for w in allowed]
        return random.choices(allowed, weights=weights, k=1)[0]

    def _can_copy_new_interval(self) -> bool:
        """World-2 is only allowed if there exists at least one generated token after the prompt."""
        return (len(self.tokens) - self.prompt_len) > 0

    def _can_copy_existing_interval(self) -> bool:
        return len(self.saved_intervals) > 0

    def _pick_interval_from_history(self) -> Tuple[int, int]:
        """
        Pick (start, length) strictly from the generated tail [prompt_len, len(tokens)).
        Length L ~ Uniform{1..min(max_window, copyable_len)}.
        Start ~ Uniform{prompt_len .. prompt_len + copyable_len - L}.
        """
        n_total = len(self.tokens)
        copyable_len = n_total - self.prompt_len
        assert copyable_len > 0, "No generated tokens to copy."

        L_max = min(self.max_window, copyable_len)
        L = random.randint(1, L_max)
        start_min = self.prompt_len
        start_max = self.prompt_len + copyable_len - L
        start = random.randint(start_min, start_max)
        return start, L

    def _copy_interval_into_tail(self, start: int, length: int):
        seg_tokens = self.tokens[start:start+length]
        seg_Us = self.Us[start:start+length]
        seg_etas = self.etas[start:start+length]
        seg_difs = self.difs[start:start+length]
        seg_maxps = self.probs[start:start+length]

        self.tokens.extend(seg_tokens)
        self.Us.extend(seg_Us)
        self.etas.extend(seg_etas)
        self.difs.extend(seg_difs)
        self.probs.extend(seg_maxps)

    # -------- worlds --------
    def _world_1_generate_one(self):
        Probs = dominate_Ps(self.Delta, self.K).astype(float)
        xi, pi = generate_uniform_local(self.tokens, self.c, self.key, self.K)
        next_token = find_next_token(xi, Probs, pi)
        eta = (pi[next_token]-1)/(self.K-1)
        highest_prob = float(np.max(Probs))

        self.tokens.append(next_token)
        self.Us.append(xi)
        self.etas.append(eta)
        self.difs.append(np.abs(xi-eta))
        self.probs.append(highest_prob)

    def _world_2_copy_new_interval(self):
        """
        Copy a NEW interval from the generated tail. If impossible, fall back to World-1.
        After copying, snapshot that interval for future World-3.
        """
        if not self._can_copy_new_interval():
            self._world_1_generate_one()
            return

        start, L = self._pick_interval_from_history()
        # copy into tail
        self._copy_interval_into_tail(start, L)
        # snapshot the just-copied segment (the last L positions now)
        self.saved_intervals.append({
            "src": "history",
            "start": start,
            "length": L,
            "tokens": self.tokens[-L:].copy(),
            "Us": self.Us[-L:].copy(),
            "etas": self.etas[-L:].copy(),
            "difs": self.difs[-L:].copy(),
            "max_probs": self.probs[-L:].copy(),
        })

    def _world_3_copy_existing_interval(self):
        """
        Copy from existing saved intervals. If none, degrade to World-2; if that also
        fails, degrade to World-1.
        """
        if not self._can_copy_existing_interval():
            if self._can_copy_new_interval():
                self._world_2_copy_new_interval()
            else:
                self._world_1_generate_one()
            return

        iv = random.choice(self.saved_intervals)
        self.tokens.extend(iv["tokens"])
        self.Us.extend(iv["Us"])
        self.etas.extend(iv["etas"])
        self.difs.extend(iv["difs"])
        self.probs.extend(iv["max_probs"])

    # -------- external API --------
    def __call__(self, prompt: List[int], steps: int = 1) -> Dict[str, List]:

        # Sync prompt into internal history (prefix)
        if len(self.tokens) < len(prompt):
            missing = len(prompt) - len(self.tokens)
            self.tokens.extend(prompt[len(self.tokens):])
            self.Us.extend([float('nan')] * missing)
            self.etas.extend([float('nan')] * missing)
            self.difs.extend([float('nan')] * missing)
            self.probs.extend([float('nan')] * missing)

        # Set prompt length once (assuming prompt never shrinks)
        if self.prompt_len == 0 and len(prompt) > 0 and len(self.tokens) >= len(prompt):
            self.prompt_len = len(prompt)

        start_size = len(self.tokens)

        # Track unique W1 tokens produced in THIS CALL (by value)
        unique_world1_tokens: set[int] = set()
        world_sequence: List[int] = []

        # Per-call counts for capping: [W1, W2, W3]
        counts = [0, 0, 0]
        rounds = 0

        # (Optional) safety cap; shouldn't trigger with capped chooser unless W1 can never add a new unique
        MAX_ROUNDS = 10**6

        while len(unique_world1_tokens) <= steps:
            rounds += 1
            if rounds > MAX_ROUNDS:
                print(
                    "Exceeded MAX_ROUNDS while waiting for unique tokens > steps. Forcing world 1"
                )
                world = 1
            else:
                world = self._choose_world_capped(counts, rounds_done=rounds - 1)

            world_sequence.append(world)
            counts[world - 1] += 1  # reserve this slot for the chosen world

            if world == 1:
                self._world_1_generate_one()
                new_tok = int(self.tokens[-1])                
                if new_tok not in unique_world1_tokens:
                    unique_world1_tokens.add(new_tok)

            elif world == 2:
                self._world_2_copy_new_interval()

            else:  # world == 3
                self._world_3_copy_existing_interval()

        # Return only what was newly appended in this call
        new_tokens = self.tokens[start_size:]
        new_Us = self.Us[start_size:]
        new_etas = self.etas[start_size:]
        new_difs = self.difs[start_size:]
        new_maxps = self.probs[start_size:]

        return {
            "new_tokens": new_tokens,
            "new_Us": new_Us,
            "new_etas": new_etas,
            "new_difs": new_difs,
            "new_highest_probs": new_maxps,
            "unique_tokens": len(unique_world1_tokens),
            "world": world_sequence,
            "counts": counts,  # how many times each world was used in this call
        }

    def summarize_prns(self, prns: Sequence[float], Us: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns three arrays aligned by index i:
        unique_prns[i] : i-th unique value of `prns` in order of first appearance (NaNs ignored)
        counts[i]      : how many times unique_prns[i] appears in `prns`
        groups[i]      : the U-group (class) of unique_prns[i], defined by the Us value at that
                        unique's first occurrence; groups are integers starting at 0 and ordered
                        by first appearance in Us (exact-equality grouping).
        """
        prns_arr = np.asarray(prns, dtype=float)
        Us_arr   = np.asarray(Us)

        if prns_arr.shape != Us_arr.shape:
            raise ValueError(f"`prns` and `Us` must have the same shape, got {prns_arr.shape} vs {Us_arr.shape}.")

        # Work only on positions where prns is not NaN
        mask = ~np.isnan(prns_arr)
        if not np.any(mask):
            return np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int)

        prns_n = prns_arr[mask]
        Us_n   = Us_arr[mask]

        # Unique PRNs by first appearance (NOT by numeric sort)
        uniq_vals, first_idx_n, counts = np.unique(prns_n, return_index=True, return_counts=True)
        # Map first indices back to original positions, then stable-sort by first appearance
        orig_pos = np.flatnonzero(mask)[first_idx_n]
        order = np.argsort(orig_pos, kind="stable")

        unique_prns = uniq_vals[order]
        counts_out  = counts[order].astype(int, copy=False)

        # Build U-groups (same Us => same group), groups ordered by first appearance in Us
        u_vals_sorted, first_u_idx, inv_sorted = np.unique(Us_n, return_index=True, return_inverse=True)
        ord_by_first = np.argsort(first_u_idx, kind="stable")
        # Map: index-in-sorted-unique-Us  ->  appearance-ordered group id
        s2g = np.empty_like(ord_by_first)
        s2g[ord_by_first] = np.arange(ord_by_first.size, dtype=int)
        # Per-position group id for each element in Us_n
        group_ids_all = s2g[inv_sorted]

        # Group for each unique PRN = group at its first occurrence; align with `order`
        groups_out = group_ids_all[first_idx_n][order].astype(int, copy=False)
        embed()
        return unique_prns, counts_out, groups_out

def generate_null_difs_with_copying(
    N_trial: int,
    final_T: int,
    K: int,
    p_types: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Null with copying that tracks per-position difs and summarizes uniques.

    Returns:
      difs                         : (N_trial, final_T)  per-position |U - eta| (with repeats)
      unique_Difs                  : (N_trial, final_T)  first final_T unique difs (order of first appearance)
      count_for_unique_difs        : (N_trial, final_T)  counts of each unique dif in the FULL row (pre-truncate)
      classes_for_unique_difs      : (N_trial, final_T)  U-group id of each unique dif (group at its first occurrence)
    """
    rng = rng or random

    difs_mat        = np.empty((N_trial, final_T), dtype=float)
    unique_difs_mat = np.empty((N_trial, final_T), dtype=float)
    counts_mat      = np.empty((N_trial, final_T), dtype=int)
    classes_mat     = np.empty((N_trial, final_T), dtype=int)

    for i in range(N_trial):
        # Full-row lists (before truncation)
        Us:   list[float] = []
        Pis:  list[int]   = []
        Etas: list[float] = []
        Difs: list[float] = []

        pool_idx: list[int] = []     # indices into the history that have been "pooled" by world-1
        uniques:  list[float] = []   # unique difs in order of first appearance
        first_pos: list[int]  = []   # first occurrence positions for each unique dif
        seen = set()                 # set of dif values we've seen

        # draw until we have > final_T unique difs
        while len(seen) <= final_T:
            world = rng.choices([0, 1, 2], weights=p_types, k=1)[0]

            if world == 0:
                # fresh
                u = rng.random()
                pi = rng.randrange(K)
                eta = (pi / (K - 1)) if K > 1 else 0.0
                d = abs(u - eta)

            elif world == 1:
                # copy from history if possible; else fresh
                if Difs:
                    idx = rng.randrange(len(Difs))
                    u, pi, eta, d = Us[idx], Pis[idx], Etas[idx], Difs[idx]
                    pool_idx.append(idx)
                else:
                    u = rng.random()
                    pi = rng.randrange(K)
                    eta = (pi / (K - 1)) if K > 1 else 0.0
                    d = abs(u - eta)

            else:  # world == 2
                # copy from pool; degrade to history -> fresh
                if pool_idx:
                    idx = pool_idx[rng.randrange(len(pool_idx))]
                    u, pi, eta, d = Us[idx], Pis[idx], Etas[idx], Difs[idx]
                elif Difs:
                    idx = rng.randrange(len(Difs))
                    u, pi, eta, d = Us[idx], Pis[idx], Etas[idx], Difs[idx]
                    pool_idx.append(idx)
                else:
                    u = rng.random()
                    pi = rng.randrange(K)
                    eta = (pi / (K - 1)) if K > 1 else 0.0
                    d = abs(u - eta)

            # append this step
            Us.append(u); Pis.append(pi); Etas.append(eta); Difs.append(d)

            if d not in seen:
                seen.add(d)
                uniques.append(d)
                first_pos.append(len(Difs) - 1)

        # truncate to final_T
        difs_trunc        = np.array(Difs[:final_T], dtype=float)
        unique_difs_trunc = np.array(uniques[:final_T], dtype=float)

        # counts of each unique dif in the FULL row (pre-truncate)
        D_full = np.asarray(Difs, dtype=float)
        counts_full = [int(np.sum(D_full == uval)) for uval in unique_difs_trunc]

        # classes: U-groups by exact equality, numbered by first U-appearance
        u_to_gid: dict[float, int] = {}
        groups_per_pos: list[int] = []
        next_gid = 0
        for u in Us:
            gid = u_to_gid.get(u)
            if gid is None:
                gid = next_gid
                u_to_gid[u] = gid
                next_gid += 1
            groups_per_pos.append(gid)

        # class for each unique dif is the group at its first occurrence
        first_pos_trunc = first_pos[:final_T]
        classes_trunc = np.array([groups_per_pos[p] for p in first_pos_trunc], dtype=int)

        # write row i
        difs_mat[i, :]        = difs_trunc
        unique_difs_mat[i, :] = unique_difs_trunc
        counts_mat[i, :]      = np.asarray(counts_full, dtype=int)
        classes_mat[i, :]     = classes_trunc

    return difs_mat, unique_difs_mat, counts_mat, classes_mat


# ---------- quick demo ----------
if __name__ == "__main__":
    K = 1000  # vocab size
    gen = Inv_repeated_token_generator(
        vocab_size=K,
        max_window=5,
        Delta=0.1,
        c=3,
        key=927472927,
        p_types=(0.9,0.05,0.05),
    )

    prompt = [11, 22, 33]
    out1 = gen(prompt, steps=10)     # likely World-1 at the very beginning
    out2 = gen(prompt, steps=50)     # now World-2/3 can kick in
    print("New tokens (first call):", len(out1["new_difs"]), "Unique:", out1["unique_tokens"],"/", len(out1["new_tokens"]))
    print("New tokens (second call):", len(out2["new_difs"]), "Unique:", out2["unique_tokens"],"/", len(out2["new_tokens"]))

    print("Unique prns  (second call):", gen.summarize_prns(out2["new_difs"], out2["new_Us"]))

    rng = random.Random(2025)  # reproducible
    difs_mat, unique_difs_mat, counts_mat, classes_mat = generate_null_difs_with_copying(N_trial=500, final_T=1000, K=K, p_types=(1/3,1/6,1/2), rng=rng)
    print("difs_mat mean:", difs_mat.mean())
    print("unique_difs_mat mean:", unique_difs_mat.mean())
