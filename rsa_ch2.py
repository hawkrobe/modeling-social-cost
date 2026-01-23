#!/usr/bin/env python3
"""
QUD-RSA model for socially-indexed referring expressions.

Extended with REGISTER dimension to differentiate TW from TGW.

Key insight: TW and TGW both signal "trans-affirming" but differ on a
vernacular/insider vs formal/clinical dimension. This matters especially
at PinkNews where signaling insider status is valued.

Four mechanisms:
  1. Semantics: constrains who can say what (epsilon parameter)
  2. Bio QUD informativity: signal trans-affirming vs bioessentialist
  3. Political QUD informativity: signal conservative/moderate/progressive
  4. Register QUD informativity: signal vernacular/insider vs formal/clinical
  5. Costs: length-based (uniform across outlets)

Usage:
    python rsa_ch2.py              # Quick comparison
    python rsa_ch2.py --ablations  # Full ablation study
    python rsa_ch2.py --info       # Show informativity breakdown

Requirements: pip install jax jaxlib scipy numpy
"""

import jax
import jax.numpy as jnp
from jax.nn import softmax
import numpy as np
import argparse
import csv

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# =============================================================================
# DOMAINS & DATA
# =============================================================================

PERSONAS = ["BB", "CJ", "BioMod", "TransMod", "TERF", "PN"]
UTTERANCES = ["biological male", "transgender woman", "trans woman"]
UTT_ABBREV = ["BM", "TGW", "TW"]

# Observed data
OBSERVED = {
    "Breitbart": jnp.array([0.449, 0.408, 0.142]),
    "PinkNews":  jnp.array([0.000, 0.241, 0.759]),
    "NPR":       jnp.array([0.000, 0.687, 0.313]),
}

# =============================================================================
# THREE QUD DIMENSIONS
# =============================================================================

# Dimension 1: Gender ideology (trans-affirming vs bioessentialist)
# Trans-affirming: CJ, TransMod, PN (indices 1, 3, 5)
# Bioessentialist: BB, BioMod, TERF (indices 0, 2, 4)
TRANS_AFF = jnp.array([0, 1, 0, 1, 0, 1])
BIO = jnp.array([1, 0, 1, 0, 1, 0])
IS_TRANS_AFF = [False, True, False, True, False, True]

# Dimension 2: Political ideology (conservative/moderate/progressive)
# Conservative: BB, CJ (indices 0, 1)
# Moderate: BioMod, TransMod (indices 2, 3)
# Progressive: TERF, PN (indices 4, 5)
CONSERVATIVE = jnp.array([1, 1, 0, 0, 0, 0])
MODERATE = jnp.array([0, 0, 1, 1, 0, 0])
PROGRESSIVE = jnp.array([0, 0, 0, 0, 1, 1])
POLITICAL_CAT = [0, 0, 1, 1, 2, 2]  # 0=con, 1=mod, 2=prog
POL_VECTORS = [CONSERVATIVE, MODERATE, PROGRESSIVE]

# Dimension 3: Register (vernacular/insider vs formal/clinical)
# Only relevant within trans-affirming space
# PN is strongly vernacular, TransMod is mixed, others don't participate
VERNACULAR = jnp.array([0.0, 0.0, 0.0, 0.3, 0.0, 1.0])  # TransMod slightly, PN strongly

# How strongly each utterance indexes vernacular register
# This is the key differentiation between TW and TGW
UTT_VERNACULAR = jnp.array([0.0, 0.2, 1.0])  # BM, TGW, TW

# =============================================================================
# SEMANTICS
# =============================================================================

def soft_compat(eps):
    """
    Semantic compatibility matrix.
    eps = probability of using a "blocked" term (eps->0: hard, eps=1: none)

    TW blocked for: BB, BioMod, TERF (bioessentialists)
    BM blocked for: CJ, TransMod, PN (trans-affirming)
    TGW: everyone can use (neutral/clinical)
    """
    return jnp.array([
        # BM   TGW   TW
        [1,    1,    eps],  # BB (bio, con)
        [eps,  1,    1  ],  # CJ (trans, con) - conservative but trans-affirming
        [1,    1,    eps],  # BioMod (bio, mod)
        [eps,  1,    1  ],  # TransMod (trans, mod)
        [1,    1,    eps],  # TERF (bio, prog)
        [eps,  1,    1  ],  # PN (trans, prog)
    ], dtype=float)

# =============================================================================
# PRIORS (hierarchical)
# =============================================================================

P_POL = {
    "Breitbart": {"con": 0.75, "mod": 0.20, "prog": 0.05},
    "PinkNews":  {"con": 0.02, "mod": 0.18, "prog": 0.80},
    "NPR":       {"con": 0.15, "mod": 0.55, "prog": 0.30},
}
P_BIO_GIVEN_POL = {"con": 0.85, "mod": 0.20, "prog": 0.05}

def hier_prior(outlet):
    """P(persona) = P(pol|outlet) × P(bio|pol)"""
    p, b = P_POL[outlet], P_BIO_GIVEN_POL
    prior = jnp.array([
        p["con"] * b["con"],       # BB
        p["con"] * (1 - b["con"]), # CJ
        p["mod"] * b["mod"],       # BioMod
        p["mod"] * (1 - b["mod"]), # TransMod
        p["prog"] * b["prog"],     # TERF
        p["prog"] * (1 - b["prog"]), # PN
    ])
    return prior / prior.sum()

PRIORS = {o: hier_prior(o) for o in OBSERVED}

# =============================================================================
# COSTS (uniform, length-based)
# =============================================================================

# Length-based costs (higher = more costly = less preferred)
# BM: "biological male" = 15 chars
# TGW: "transgender woman" = 17 chars
# TW: "trans woman" = 11 chars
COSTS_UNIFORM = jnp.array([0.5, 0.6, 0.0])  # TW cheapest, TGW most expensive, BM middle

COSTS = {
    "Breitbart": COSTS_UNIFORM,
    "PinkNews":  COSTS_UNIFORM,
    "NPR":       COSTS_UNIFORM,
}

# =============================================================================
# MODEL
# =============================================================================

@jax.jit
def predict_jit(prior, alpha, bio_w, pol_w, reg_w, cost_w, eps):
    """
    Vectorized prediction using JAX.
    """
    compat = soft_compat(eps)  # (6, 3)

    # L0 posteriors for all utterances: P(persona | utterance)
    # prior: (6,), compat: (6, 3) -> posteriors: (6, 3)
    posteriors = prior[:, None] * compat  # (6, 3)
    posteriors = posteriors / (posteriors.sum(axis=0, keepdims=True) + 1e-10)  # normalize per utterance

    # Bio informativity for each utterance
    p_trans_given_u = jnp.sum(posteriors * TRANS_AFF[:, None], axis=0)  # (3,)
    p_bio_given_u = jnp.sum(posteriors * BIO[:, None], axis=0)  # (3,)

    # Political informativity for each utterance and political category
    p_con_given_u = jnp.sum(posteriors * CONSERVATIVE[:, None], axis=0)  # (3,)
    p_mod_given_u = jnp.sum(posteriors * MODERATE[:, None], axis=0)  # (3,)
    p_prog_given_u = jnp.sum(posteriors * PROGRESSIVE[:, None], axis=0)  # (3,)

    # Register: P(vernacular | utterance)
    p_vern_given_u = jnp.sum(posteriors * VERNACULAR[:, None], axis=0)  # (3,)

    # Now compute utilities for each persona
    production = jnp.zeros(3)

    for p_idx in range(6):
        is_trans_aff = IS_TRANS_AFF[p_idx]
        pol_cat = POLITICAL_CAT[p_idx]
        persona_vern = VERNACULAR[p_idx]

        # Bio info: pick trans or bio based on persona
        bio_prob = jnp.where(is_trans_aff, p_trans_given_u, p_bio_given_u)
        bio_info = jnp.log(bio_prob + 1e-10)

        # Political info: pick the right category
        pol_prob = jnp.where(pol_cat == 0, p_con_given_u,
                            jnp.where(pol_cat == 1, p_mod_given_u, p_prog_given_u))
        pol_info = jnp.log(pol_prob + 1e-10)

        # Register info (only for trans-affirming vernacular personas)
        reg_info = jnp.where(
            is_trans_aff & (persona_vern > 0),
            persona_vern * UTT_VERNACULAR * jnp.log(p_vern_given_u + 1e-10),
            0.0
        )

        # Utility
        utilities = bio_w * bio_info + pol_w * pol_info + reg_w * reg_info - cost_w * COSTS_UNIFORM

        # S1 with semantic constraints
        s1_logits = alpha * utilities + jnp.log(compat[p_idx] + 1e-10)
        s1 = softmax(s1_logits)

        production = production + prior[p_idx] * s1

    return production

def predict(outlet, alpha, bio_w, pol_w, reg_w, cost_w, eps):
    """Wrapper for JIT-compiled prediction."""
    prior = PRIORS[outlet]
    return np.array(predict_jit(prior, alpha, bio_w, pol_w, reg_w, cost_w, eps))

def rmse(pred, obs):
    return float(np.sqrt(np.mean((np.array(pred) - np.array(obs))**2)))

def total_rmse(params):
    """Compute average RMSE across all outlets."""
    alpha, bio_w, pol_w, reg_w, cost_w, eps = params
    total = 0
    for outlet in OBSERVED:
        pred = predict(outlet, alpha, bio_w, pol_w, reg_w, cost_w, eps)
        total += rmse(pred, OBSERVED[outlet])
    return total / len(OBSERVED)

# =============================================================================
# FITTING
# =============================================================================

def fit_model(use_bio=True, use_pol=True, use_reg=True, use_cost=True, n_starts=5):
    """Pure scipy optimize with multiple random starts."""
    from scipy.optimize import minimize

    def objective(x):
        return total_rmse(tuple(x))

    bounds = [(0.1, 20), (0, 50), (0, 50), (0, 50), (0, 50), (0.0001, 0.2)]
    if not use_bio: bounds[1] = (0, 0)
    if not use_pol: bounds[2] = (0, 0)
    if not use_reg: bounds[3] = (0, 0)
    if not use_cost: bounds[4] = (0, 0)

    best_rmse = float('inf')
    best_x = None

    # Multiple starting points
    np.random.seed(42)
    for _ in range(n_starts):
        x0 = [
            np.random.uniform(1, 5),      # alpha
            np.random.uniform(0, 5) if use_bio else 0,
            np.random.uniform(0, 5) if use_pol else 0,
            np.random.uniform(0, 5) if use_reg else 0,
            np.random.uniform(0, 5) if use_cost else 0,
            np.random.uniform(0.001, 0.1)  # eps
        ]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200})
        if result.fun < best_rmse:
            best_rmse = result.fun
            best_x = result.x

    return best_rmse, {
        'alpha': best_x[0],
        'bio_w': best_x[1],
        'pol_w': best_x[2],
        'reg_w': best_x[3],
        'cost_w': best_x[4],
        'eps': best_x[5]
    }

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compare():
    """Compare models with and without register dimension."""
    print("\n" + "="*75)
    print("MODEL COMPARISON: Does Register dimension help?")
    print("="*75)

    configs = [
        # (bio, pol, reg, cost, name)
        (True,  False, False, False, "Bio only"),
        (False, True,  False, False, "Pol only"),
        (False, False, True,  False, "Reg only"),
        (False, False, False, True,  "Cost only"),
        (True,  True,  False, False, "Bio + Pol"),
        (True,  True,  False, True,  "Bio + Pol + Cost"),
        (True,  True,  True,  False, "Bio + Pol + Reg"),
        (True,  True,  True,  True,  "Full (Bio+Pol+Reg+Cost)"),
    ]

    results = []
    for use_bio, use_pol, use_reg, use_cost, name in configs:
        avg_rmse, params = fit_model(use_bio, use_pol, use_reg, use_cost)
        results.append((avg_rmse, name, params, use_reg))

    results.sort(key=lambda x: x[0])

    print(f"\n{'Model':<25} {'RMSE':>7} {'bio':>5} {'pol':>5} {'reg':>5} {'cost':>5} {'α':>4} {'ε':>5}")
    print("-"*75)
    for rmse_val, name, params, has_reg in results:
        marker = "***" if has_reg else "   "
        print(f"{name:<25} {rmse_val:>7.4f} {params['bio_w']:>5.2f} {params['pol_w']:>5.2f} "
              f"{params['reg_w']:>5.2f} {params['cost_w']:>5.2f} {params['alpha']:>4.1f} {params['eps']:>5.2f} {marker}")

    # Show predictions for best model
    best_rmse, best_name, best_params, _ = results[0]
    print(f"\n{'='*75}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*75}")

    for outlet in ["Breitbart", "NPR", "PinkNews"]:
        pred = predict(outlet, **best_params)
        obs = OBSERVED[outlet]
        r = rmse(pred, obs)
        print(f"\n{outlet}:")
        print(f"  Predicted: BM={pred[0]:.3f}  TGW={pred[1]:.3f}  TW={pred[2]:.3f}")
        print(f"  Observed:  BM={float(obs[0]):.3f}  TGW={float(obs[1]):.3f}  TW={float(obs[2]):.3f}")
        print(f"  RMSE: {r:.4f}")

    # Compare with and without register for the full model
    print(f"\n{'='*75}")
    print("KEY COMPARISON: Full model WITH vs WITHOUT register")
    print(f"{'='*75}")

    _, params_no_reg = fit_model(True, True, False, True)
    _, params_with_reg = fit_model(True, True, True, True)

    for outlet in ["Breitbart", "NPR", "PinkNews"]:
        pred_no = predict(outlet, **params_no_reg)
        pred_yes = predict(outlet, **params_with_reg)
        obs = OBSERVED[outlet]

        print(f"\n{outlet}:")
        print(f"  Without Reg: BM={pred_no[0]:.3f}  TGW={pred_no[1]:.3f}  TW={pred_no[2]:.3f}  RMSE={rmse(pred_no, obs):.4f}")
        print(f"  With Reg:    BM={pred_yes[0]:.3f}  TGW={pred_yes[1]:.3f}  TW={pred_yes[2]:.3f}  RMSE={rmse(pred_yes, obs):.4f}")
        print(f"  Observed:    BM={float(obs[0]):.3f}  TGW={float(obs[1]):.3f}  TW={float(obs[2]):.3f}")

def show_informativity():
    """Show how each utterance signals each dimension at each outlet."""
    print("\n" + "="*75)
    print("INFORMATIVITY BREAKDOWN BY OUTLET")
    print("="*75)

    eps = 0.05
    compat = soft_compat(eps)

    for outlet in ["Breitbart", "NPR", "PinkNews"]:
        prior = PRIORS[outlet]
        print(f"\n{outlet}")
        print(f"  Prior: " + " ".join(f"{PERSONAS[i]}={float(prior[i]):.2f}" for i in range(6)))
        print("-"*60)

        for u_idx, utt in enumerate(UTTERANCES):
            post = prior * compat[:, u_idx]
            post = post / post.sum()

            # Bio dimension
            p_trans = float(jnp.sum(post * TRANS_AFF))
            p_bio = float(jnp.sum(post * BIO))

            # Political dimension
            p_con = float(jnp.sum(post * CONSERVATIVE))
            p_mod = float(jnp.sum(post * MODERATE))
            p_prog = float(jnp.sum(post * PROGRESSIVE))

            # Register dimension
            p_vern = float(jnp.sum(post * VERNACULAR))
            utt_vern = float(UTT_VERNACULAR[u_idx])

            print(f"  {UTT_ABBREV[u_idx]:>3}: bio[trans={p_trans:.2f}] "
                  f"pol[c={p_con:.2f} m={p_mod:.2f} p={p_prog:.2f}] "
                  f"reg[vern={p_vern:.2f}, utt_sig={utt_vern:.1f}]")

def run_ablations(outpath="ablations.csv"):
    """Full ablation study saved to CSV."""
    configs = [
        (True,  False, False, False, "Bio"),
        (False, True,  False, False, "Pol"),
        (False, False, True,  False, "Reg"),
        (False, False, False, True,  "Cost"),
        (True,  True,  False, False, "Bio+Pol"),
        (True,  False, True,  False, "Bio+Reg"),
        (True,  False, False, True,  "Bio+Cost"),
        (False, True,  True,  False, "Pol+Reg"),
        (False, True,  False, True,  "Pol+Cost"),
        (False, False, True,  True,  "Reg+Cost"),
        (True,  True,  True,  False, "Bio+Pol+Reg"),
        (True,  True,  False, True,  "Bio+Pol+Cost"),
        (True,  False, True,  True,  "Bio+Reg+Cost"),
        (False, True,  True,  True,  "Pol+Reg+Cost"),
        (True,  True,  True,  True,  "Full"),
    ]

    results = []
    for use_bio, use_pol, use_reg, use_cost, name in configs:
        avg_rmse, params = fit_model(use_bio, use_pol, use_reg, use_cost)

        for outlet in OBSERVED:
            pred = predict(outlet, **params)
            obs = OBSERVED[outlet]
            r = rmse(pred, obs)

            results.append({
                "model": name,
                "bio": "yes" if use_bio else "no",
                "pol": "yes" if use_pol else "no",
                "reg": "yes" if use_reg else "no",
                "cost": "yes" if use_cost else "no",
                "alpha": params['alpha'],
                "bio_w": params['bio_w'],
                "pol_w": params['pol_w'],
                "reg_w": params['reg_w'],
                "cost_w": params['cost_w'],
                "epsilon": params['eps'],
                "outlet": outlet,
                "pred_bm": float(pred[0]),
                "pred_tgw": float(pred[1]),
                "pred_tw": float(pred[2]),
                "obs_bm": float(obs[0]),
                "obs_tgw": float(obs[1]),
                "obs_tw": float(obs[2]),
                "rmse": r,
                "avg_rmse": avg_rmse,
            })

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} rows to {outpath}")
    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RSA Model with Register Dimension")
    parser.add_argument("--compare", action="store_true", help="Compare model variants (default)")
    parser.add_argument("--ablations", type=str, nargs="?", const="ablations.csv",
                        help="Save ablation results to CSV")
    parser.add_argument("--info", action="store_true", help="Show informativity breakdown")
    args = parser.parse_args()

    if args.ablations:
        run_ablations(args.ablations)
    elif args.info:
        show_informativity()
    else:
        compare()

if __name__ == "__main__":
    main()
