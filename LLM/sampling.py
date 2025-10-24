import torch
from alternative_prf_schemes import prf_lookup

def seed_rng(generator, tokens, seeding_scheme="minhash_prf", hash_key=15485863, c=5):
    """
    Seed the random number generator using a context-based PRF hash.

    Args:
        generator: PyTorch random generator.
        tokens (Tensor): Input tokens with shape (1, current_length).
        seeding_scheme (str): PRF type, e.g., "minhash_prf".
        hash_key (int): Salt for the PRF.
        c (int): Number of context tokens used for hashing.
    """
    assert tokens.shape[-1] >= c, f"seeding_scheme={seeding_scheme} requires at least {c} context tokens"
    prf_key = prf_lookup[seeding_scheme](tokens[0][-c:], salt_key=hash_key)
    generator.manual_seed(prf_key)

######################################
# Gumbel-Max Watermarking
######################################

def gumbel_key_func(generator, inputs, vocab_size, key, c, seeding_scheme):
    """
    Generate Gumbel noise xi and identity permutation for Gumbel-Max watermarking.

    Returns:
        xis: Tensor of shape (batch_size, vocab_size)
        pis: Tensor of shape (batch_size, vocab_size)
    """
    xis = []
    pis = []
    for k in range(inputs.shape[0]):
        seed_rng(generator, inputs[k].unsqueeze(0), seeding_scheme=seeding_scheme, hash_key=key, c=c)
        xi = torch.rand(size=(1, vocab_size), generator=generator)
        pi = torch.arange(vocab_size)
        xis.append(xi)
        pis.append(pi)
    return torch.vstack(xis), torch.vstack(pis)

def gumbel_sampling(probs, pi, xi):
    """
    Sample next token index via Gumbel-Max trick.
    """
    return torch.argmax(xi ** (1 / torch.gather(probs, 1, pi)), axis=1).unsqueeze(-1)

def gumbel_Y(s, pi, xi):
    """
    Retrieve the Gumbel noise value corresponding to the sampled token.
    """
    return torch.gather(xi, -1, s.cpu()).squeeze()

######################################
# Inverse-Transform Watermarking
######################################

def transform_key_func(generator, inputs, vocab_size, key, c, seeding_scheme):
    """
    Generate xi and pi for inverse-transform watermarking (batch-supported).
    Returns:
        xi: (batch_size, 1)
        pi: (batch_size, vocab_size)
    """
    batch_size = inputs.shape[0]
    xis = torch.rand((batch_size, 1), generator=generator)

    # Randperm isn't batched in PyTorch, so we use a loop — fast, and batch-friendly
    pis = torch.stack([torch.randperm(vocab_size, generator=generator) for _ in range(batch_size)])
    return xis, pis


def batch_inverse_permutation(pis):
    """
    Batched inverse permutation for a (batch_size, vocab_size) tensor.
    """
    inv = torch.empty_like(pis)
    arange = torch.arange(pis.shape[1], device=pis.device).expand_as(pis)
    inv.scatter_(1, pis, arange)
    return inv


def transform_sampling(probs, pi, xi):
    """
    Perform inverse transform sampling using CDF and sorted permutation (batched).
    Inputs:
        probs: (batch_size, vocab_size)
        pi: (batch_size, vocab_size)
        xi: (batch_size, 1)
    Output:
        token indices: (batch_size, 1)
    """
    inv_pi = batch_inverse_permutation(pi)  # (batch_size, vocab_size)
    gathered_probs = torch.gather(probs, 1, inv_pi)
    cdf = torch.cumsum(gathered_probs, dim=1)
    indices = torch.searchsorted(cdf, xi, right=False)
    return torch.gather(inv_pi, 1, indices)


def transform_Y_dif(s, pi, xi):
    """
    Compute distance score between xi and rank-normalized token under permutation.
    Inputs:
        s: (batch_size, 1)a
        pi: (batch_size, vocab_size)
        xi: (batch_size, 1)
    Output:
        score: (batch_size,)
    """
    vocab_size = pi.shape[1]
    # Make sure s is on the same device as pi
    s = s.to(pi.device)
    xi = xi.to(pi.device)
    s_rank = torch.gather(pi, 1, s)
    quantile = (s_rank - 1) / (vocab_size - 1)
    return -torch.abs(xi - quantile).squeeze(1)
