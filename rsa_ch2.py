#!/usr/bin/env python3
"""
QUD-RSA model for socially-indexed referring expressions using memo.

Three components:
  1. Semantics: who can say what (with register built in)
  2. Informativity: speakers want to be correctly identified
  3. Costs: length-based

Usage:
    python rsa_ch2.py
    python rsa_ch2.py --ablations ablations.csv
"""

from memo import memo
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import argparse
import csv

jax.config.update("jax_enable_x64", True)

# =============================================================================
# DOMAINS
# =============================================================================

P = jnp.arange(6)  # personas: BB, CJ, BioMod, TransMod, TERF, PN
U = jnp.arange(3)  # utterances: BM, TGW, TW

PERSONA_NAMES = ["BB", "CJ", "BioMod", "TransMod", "TERF", "PN"]
UTT_NAMES = ["BM", "TGW", "TW"]

OBSERVED = {
    "Breitbart": jnp.array([0.449, 0.408, 0.142]),
    "PinkNews":  jnp.array([0.000, 0.241, 0.759]),
    "NPR":       jnp.array([0.000, 0.687, 0.313]),
}

# =============================================================================
# PRIORS
# =============================================================================

P_POL = {
    "Breitbart": {"con": 0.75, "mod": 0.20, "prog": 0.05},
    "PinkNews":  {"con": 0.02, "mod": 0.18, "prog": 0.80},
    "NPR":       {"con": 0.15, "mod": 0.55, "prog": 0.30},
}
P_BIO_GIVEN_POL = {"con": 0.85, "mod": 0.20, "prog": 0.05}

def make_prior(outlet):
    p, b = P_POL[outlet], P_BIO_GIVEN_POL
    prior = jnp.array([
        p["con"] * b["con"],         # BB
        p["con"] * (1 - b["con"]),   # CJ
        p["mod"] * b["mod"],         # BioMod
        p["mod"] * (1 - b["mod"]),   # TransMod
        p["prog"] * b["prog"],       # TERF
        p["prog"] * (1 - b["prog"]), # PN
    ])
    return prior / prior.sum()

PRIORS = {o: make_prior(o) for o in OBSERVED}

# =============================================================================
# SEMANTICS & COSTS
# =============================================================================

@jax.jit
def compat(p, u, eps, insider_eps):
    """
    Semantic compatibility: can persona p say utterance u?

    - BM blocked for trans-affirming (eps)
    - TW blocked for bioessentialist (eps)
    - TW restricted for non-insider trans-affirming (insider_eps)
    - TGW available to everyone
    """
    # Build compatibility matrix
    mat = jnp.array([
        # BM   TGW   TW
        [1,    1,    eps],         # BB (bio)
        [eps,  1,    insider_eps], # CJ (trans, not insider)
        [1,    1,    eps],         # BioMod (bio)
        [eps,  1,    insider_eps], # TransMod (trans, not insider)
        [1,    1,    eps],         # TERF (bio)
        [eps,  1,    1],           # PN (trans, insider)
    ])
    return mat[p, u]

COSTS = jnp.array([0.5, 0.6, 0.0])  # BM, TGW, TW (length-based)

@jax.jit
def cost(u):
    return COSTS[u]

# =============================================================================
# RSA MODEL (using memo)
# =============================================================================

@jax.jit
def prior_wpp(p, prior):
    return prior[p]

@memo
def S1[p: P, u: U](alpha, cost_w, eps, insider_eps, prior: ...):
    """S1 speaker: choose utterance to be correctly identified."""
    speaker: knows(p)
    speaker: thinks[
        listener: thinks[
            speaker: given(p in P, wpp=prior_wpp(p, prior)),
            speaker: chooses(u in U, wpp=compat(p, u, eps, insider_eps))
        ]
    ]
    speaker: chooses(u in U, wpp=exp(alpha * imagine[
        listener: observes [speaker.u] is u,
        listener: knows(p),
        (
            listener[ log(Pr[speaker.p == p]) ] -  # informativity
            cost_w * cost(u)                        # cost
        )
    ]))
    return Pr[speaker.u == u]

# =============================================================================
# PREDICTION
# =============================================================================

def predict(outlet, alpha, cost_w, eps, insider_eps):
    """Predict utterance distribution at an outlet."""
    prior = PRIORS[outlet]
    s1 = S1(alpha, cost_w, eps, insider_eps, prior=prior)
    # Marginalize over personas
    production = jnp.sum(prior[:, None] * s1, axis=0)
    return np.array(production)

def rmse(pred, obs):
    return float(np.sqrt(np.mean((np.array(pred) - np.array(obs))**2)))

def total_rmse(params):
    alpha, cost_w, eps, insider_eps = params
    total = 0
    for outlet in OBSERVED:
        pred = predict(outlet, alpha, cost_w, eps, insider_eps)
        total += rmse(pred, OBSERVED[outlet])
    return total / len(OBSERVED)

# =============================================================================
# FITTING
# =============================================================================

def fit_model(use_insider=True, n_starts=5):
    """Fit with scipy optimize."""
    bounds = [(0.1, 20), (0, 20), (0.0001, 0.3),
              (0.0001, 1.0) if use_insider else (1.0, 1.0)]

    best_rmse = float('inf')
    best_x = None

    np.random.seed(42)
    for _ in range(n_starts):
        x0 = [
            np.random.uniform(1, 5),
            np.random.uniform(0, 5),
            np.random.uniform(0.001, 0.1),
            np.random.uniform(0.1, 0.5) if use_insider else 1.0,
        ]
        result = minimize(total_rmse, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200})
        if result.fun < best_rmse:
            best_rmse = result.fun
            best_x = result.x

    return best_rmse, {
        'alpha': best_x[0],
        'cost_w': best_x[1],
        'eps': best_x[2],
        'insider_eps': best_x[3]
    }

# =============================================================================
# ABLATIONS
# =============================================================================

def run_ablations(outpath="ablations.csv"):
    """Compare model with vs without register."""
    results = []

    for use_insider, name in [(False, "No Register"), (True, "With Register")]:
        avg_rmse, params = fit_model(use_insider=use_insider)

        for outlet in OBSERVED:
            pred = predict(outlet, **params)
            obs = OBSERVED[outlet]
            r = rmse(pred, obs)

            results.append({
                "model": name,
                "alpha": params['alpha'],
                "cost_w": params['cost_w'],
                "eps": params['eps'],
                "insider_eps": params['insider_eps'],
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

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", type=str, nargs="?", const="ablations.csv")
    args = parser.parse_args()

    if args.ablations:
        run_ablations(args.ablations)
        return

    print("\n" + "="*65)
    print("QUD-RSA: Semantics + Informativity + Cost")
    print("="*65)

    for use_insider, name in [(False, "Without Register"), (True, "With Register")]:
        avg_rmse, params = fit_model(use_insider=use_insider)

        print(f"\n{name}:")
        print(f"  α={params['alpha']:.2f}, cost_w={params['cost_w']:.2f}, "
              f"ε={params['eps']:.4f}, insider_ε={params['insider_eps']:.4f}")
        print(f"  Avg RMSE: {avg_rmse:.4f}")

        for outlet in ["Breitbart", "NPR", "PinkNews"]:
            pred = predict(outlet, **params)
            obs = OBSERVED[outlet]
            r = rmse(pred, obs)
            print(f"    {outlet}: pred=[{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}] "
                  f"obs=[{float(obs[0]):.3f}, {float(obs[1]):.3f}, {float(obs[2]):.3f}] "
                  f"RMSE={r:.4f}")

if __name__ == "__main__":
    main()
