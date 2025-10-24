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
def generate_uniform_local(inputs: List[int], c: int, key: int, K: int) -> np.ndarray:
    """
    Generate a reproducible U(0,1)^K vector based on the last (c-1) tokens and a key.
    If not enough tokens yet, use what we have.
    """
    tail = inputs[-(c-1):] if c > 1 else []
    random.seed(tuple(tail + [key]))
    np.random.seed(random.randrange(2**32 - 1))
    return np.random.uniform(size=K)

# ---------- the generator ----------
class Gumbel_repeated_token_generator:
    """
    Three worlds per step:
      1) Generate 1 new token with your probabilistic rule.
      2) Copy a NEW interval from the already-generated tail (excluding the prompt),
         interval length L in [1, min(max_window, generated_len)], start chosen uniformly.
         The copied interval (tokens, selected_xis, max_probs) is also SNAPSHOTTED
         into saved_intervals for future reuse.
      3) Copy an EXISTING interval from saved_intervals (snapshot), chosen uniformly.

    State tracked:
      - self.tokens:    full token history (prompt prefix + generated tokens)
      - self.xis:       per token, the selected xi (float); NaN for prompt prefix
      - self.max_probs: per token, the highest prob in Probs; NaN for prompt prefix
      - self.saved_intervals: interval snapshots recorded in World-2
      - self.prompt_len: length of the (latest) prompt prefix; World-2 can only copy from [prompt_len, ...)
    """
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
        self.xis: List[float] = []
        self.max_probs: List[float] = []
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
        seg_xis = self.xis[start:start+length]
        seg_maxps = self.max_probs[start:start+length]
        self.tokens.extend(seg_tokens)
        self.xis.extend(seg_xis)
        self.max_probs.extend(seg_maxps)

    # -------- worlds --------
    def _world_1_generate_one(self):
        """
        Generate a single new token using your rule:
          Probs = uniform_Ps(Delta), U = generate_uniform_local(...),
          next_token = argmax(U ** (1 / Probs)).
        """
        Probs = dominate_Ps(self.Delta, self.K).astype(float)
        U = generate_uniform_local(self.tokens, self.c, self.key, self.K)

        next_token = int(np.argmax(np.power(U, 1.0 / np.maximum(Probs, 1e-12))))
        selected_xi = float(U[next_token])
        highest_prob = float(np.max(Probs))

        self.tokens.append(next_token)
        self.xis.append(selected_xi)
        self.max_probs.append(highest_prob)

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
            "xis": self.xis[-L:].copy(),
            "max_probs": self.max_probs[-L:].copy(),
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
        self.xis.extend(iv["xis"])
        self.max_probs.extend(iv["max_probs"])

    # -------- external API --------
    def __call__(self, prompt: List[int], steps: int = 1) -> Dict[str, List]:
        # Sync prompt into internal history (prefix)
        if len(self.tokens) < len(prompt):
            missing = len(prompt) - len(self.tokens)
            self.tokens.extend(prompt[len(self.tokens):])
            self.xis.extend([float('nan')] * missing)
            self.max_probs.extend([float('nan')] * missing)

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
        new_xis = self.xis[start_size:]
        new_maxps = self.max_probs[start_size:]
        return {
            "new_tokens": new_tokens,
            "new_selected_xis": new_xis,
            "new_highest_probs": new_maxps,
            "unique_tokens": len(unique_world1_tokens),
            "world": world_sequence,
            "counts": counts,  # how many times each world was used in this call
        }

    
    def summarize_prns(self, prns: Sequence[float]) -> Tuple[List[float], List[int]]:
        """
        Given an array of pseudorandom numbers `prns`, return:
          - unique_prns: unique values in order of first appearance
          - counts: how many times each unique value appears
        Notes:
          - NaN values are ignored.
          - Equality is exact (no tolerance), which matches your copy behavior.
        """
        arr = np.asarray(prns, dtype=float)
        uniq, first_idx, counts = np.unique(arr, return_index=True, return_counts=True)
        order = np.argsort(first_idx, kind="stable")
        return uniq[order], counts[order]


import numpy as np
import random
from typing import Tuple, Optional

def generate_null_with_copying(
    N_trial: int,
    final_T: int,
    p_types: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each trial (row):
      - Keep sampling values according to three types until the number of unique
        values in the row is > final_T.
        Types:
          0: with prob p_types[0], append a fresh U(0,1)
          1: with prob p_types[1], copy uniformly from history (the row built so far)
             and also append that value to the copied_pool
          2: with prob p_types[2], sample uniformly from copied_pool;
             if copied_pool is empty, fall back to type 1; if history empty, fall back to fresh
      - Let A be the resulting list (length will be > final_T).
      - Output three arrays of length final_T each:
          A_trunc  = A[:final_T]                       (may contain repeats)
          U_trunc  = first final_T unique values seen  (order of first appearance)
          C_trunc  = counts of each unique value in A_trunc, aligned with U_trunc

    Returns:
      A_mat: np.ndarray of shape (N_trial, final_T)
      U_mat: np.ndarray of shape (N_trial, final_T)
      C_mat: np.ndarray of shape (N_trial, final_T)
    """
    rng = rng or random

    A_mat = np.empty((N_trial, final_T), dtype=float)
    U_mat = np.empty((N_trial, final_T), dtype=float)
    C_mat = np.empty((N_trial, final_T), dtype=int)

    for i in range(N_trial):
        A = []                # the growing row (with repeats)
        copied_pool = []      # values copied via type-1 to enable type-2
        uniques = []          # uniques in order of first appearance
        seen = set()

        # keep sampling until unique count > final_T
        while len(seen) <= final_T:
            world = rng.choices([0, 1, 2], weights=p_types, k=1)[0]

            if world == 0:
                # fresh U(0,1)
                x = rng.random()

            elif world == 1:
                # copy from history if possible; else fresh
                if A:
                    x = A[rng.randrange(len(A))]
                    copied_pool.append(x)
                else:
                    x = rng.random()

            else:  # world == 2
                # sample from copied_pool; degrade to history -> fresh if needed
                if copied_pool:
                    x = copied_pool[rng.randrange(len(copied_pool))]
                elif A:
                    x = A[rng.randrange(len(A))]
                    copied_pool.append(x)
                else:
                    x = rng.random()

            A.append(x)
            if x not in seen:
                seen.add(x)
                uniques.append(x)

        # truncate as requested (both arrays length = final_T)
        A_trunc = np.array(A[:final_T], dtype=float)
        U_trunc = np.array(uniques[:final_T], dtype=float)

        # count occurrences of each unique in A_trunc
        A_full = np.array(A, dtype=float)
        counts_full = [int(np.sum(A_full == u)) for u in U_trunc]

        A_mat[i, :] = A_trunc
        U_mat[i, :] = U_trunc
        C_mat[i, :] = np.array(counts_full, dtype=int)

    return A_mat, U_mat, C_mat




# ---------- quick demo ----------
if __name__ == "__main__":
    K = 1000  # vocab size
    gen = Gumbel_repeated_token_generator(
        vocab_size=K,
        max_window=4,
        Delta=0.1,
        c=4,
        key=927472927,
        p_types=(1/2,1/3,1/6),
    )

    prompt = [11, 22, 33]
    out1 = gen(prompt, steps=10)     # likely World-1 at the very beginning
    out2 = gen(prompt, steps=50)     # now World-2/3 can kick in
    print("New tokens (first call):", out1["new_tokens"], "Unique:", out1["unique_tokens"],"/", len(out1["new_tokens"]))
    print("New tokens (second call):", out2["new_tokens"][:20], "Unique:", out2["unique_tokens"],"/", len(out2["new_tokens"]))
    print("Unique prns  (second call):", len(gen.summarize_prns(out2["new_tokens"])[0]))

    rng = random.Random(2025)  # reproducible
    A_mat, U_mat = generate_null_with_copying(N_trial=500, final_T=1000, p_types=(1/3,1/6,1/2), rng=rng)
    print("A_mat mean:", A_mat.mean())
    print("U_mat mean:", U_mat.mean())
