#!/usr/bin/env python3
"""
Dual-pathway RSA model for socially-indexed referring expressions.

Usage:
    python rsa_ch2.py --compare     # Compare model variants
    python rsa_ch2.py               # Run full model on all outlets

Requirements: pip install memo-lang jax jaxlib
"""

from memo import memo
from enum import IntEnum
import jax
import jax.numpy as jnp
import argparse

# =============================================================================
# DOMAINS & DATA
# =============================================================================

class Persona(IntEnum):
    BB = 0        # Conservative + Bioessentialist
    CJ = 1        # Conservative + Trans-affirming
    BIO_MOD = 2   # Moderate + Bioessentialist
    TRANS_MOD = 3 # Moderate + Trans-affirming
    TERF = 4      # Progressive + Bioessentialist
    PN = 5        # Progressive + Trans-affirming

class Utterance(IntEnum):
    BIO_MALE = 0  # "biological male"
    TG_WOMAN = 1  # "transgender woman"
    TRANS_W = 2   # "trans woman"

UTT_NAMES = ["biological male", "transgender woman", "trans woman"]

OBSERVED = {
    "Breitbart": jnp.array([0.449, 0.408, 0.142]),
    "PinkNews":  jnp.array([0.000, 0.241, 0.759]),
    "NPR":       jnp.array([0.000, 0.687, 0.313]),
}

# Semantic compatibility: who can use what
# "biological male" → conservative OR bioessentialist
# "trans woman" → trans-affirming (not bioessentialist)
# "transgender woman" → anyone
COMPAT = jnp.array([
    [1, 1, 0], [1, 1, 1], [1, 1, 0],  # BB, CJ, BioMod
    [0, 1, 1], [1, 1, 0], [0, 1, 1],  # TransMod, TERF, PN
], dtype=float)

# =============================================================================
# PRIORS
# =============================================================================

# Flat priors: 6 independent parameters per outlet (hand-specified)
def normalize(x): return x / x.sum()
FLAT_PRIORS = {
    "Breitbart": normalize(jnp.array([.60, .15, .13, .10, .01, .01])),
    "PinkNews":  normalize(jnp.array([.01, .01, .10, .25, .13, .60])),
    "NPR":       normalize(jnp.array([.10, .10, .15, .45, .15, .15])),
}

# Hierarchical: P(persona) = P(pol) × P(bio|pol)
P_POL = {
    "Breitbart": {"con": .75, "mod": .20, "prog": .05},
    "PinkNews":  {"con": .02, "mod": .18, "prog": .80},
    "NPR":       {"con": .15, "mod": .55, "prog": .30},
}
P_BIO_GIVEN_POL = {"con": .85, "mod": .40, "prog": .15}

def hier_prior(outlet):
    p, b = P_POL[outlet], P_BIO_GIVEN_POL
    prior = jnp.array([
        p["con"]*b["con"], p["con"]*(1-b["con"]),
        p["mod"]*b["mod"], p["mod"]*(1-b["mod"]),
        p["prog"]*b["prog"], p["prog"]*(1-b["prog"]),
    ])
    return prior / prior.sum()

HIER_PRIORS = {o: hier_prior(o) for o in OBSERVED}

# =============================================================================
# COSTS (more negative = more accessible)
# =============================================================================

COST_GENERAL = jnp.array([-12.05, -14.62, -14.85])
COST_OUTLET = {
    "Breitbart": jnp.array([-16., -14., -12.]),  # BM accessible
    "PinkNews":  jnp.array([-10., -14., -17.]),  # TW accessible
    "NPR":       jnp.array([-11., -16., -14.]),  # TGW accessible
}

# =============================================================================
# MODEL
# =============================================================================

ALPHA, SOCIAL_W, COST_W = 1.0, 0.5, 0.5

@jax.jit
def idx(arr, i): return arr[i]

@jax.jit
def idx2(arr, i, j): return arr[i, j]

@memo
def S1[p: Persona, u: Utterance](alpha, social_w, cost_w, prior: ..., costs: ..., compat: ...):
    """
    Pragmatic speaker: P(utterance | persona)

    Speaker knows their persona and chooses utterance to maximize:
      U(u,p) = social_w × log L0(p|u) - cost_w × cost(u)
    """
    speaker: knows(p)
    speaker: thinks[
        # Speaker models literal listener
        listener: thinks[
            spk: chooses(pers in Persona, wpp=idx(prior, pers)),
            spk: chooses(utt in Utterance, wpp=idx2(compat, pers, utt))
        ]
    ]
    speaker: chooses(utt in Utterance, wpp=idx2(compat, p, utt) * exp(alpha * imagine[
        listener: observes [spk.utt] is utt,
        listener: knows(p),
        (
            social_w * listener[ log(Pr[spk.pers == p] + 1e-10) ] -  # Informativity
            cost_w * idx(costs, utt)                                  # Cost
        )
    ]))
    return Pr[speaker.utt == u]

@memo
def production[u_query: Utterance](alpha, social_w, cost_w, prior: ..., costs: ..., compat: ...):
    """Expected production: P(utt) = Σ_p P(p) × S1(utt|p)"""
    world: chooses(pers in Persona, wpp=idx(prior, pers))
    world: chooses(utt in Utterance, wpp=S1[pers, utt](alpha, social_w, cost_w, prior, costs, compat))
    return Pr[world.utt == u_query]

def run(prior, costs):
    """Compute production distribution"""
    return production(ALPHA, SOCIAL_W, COST_W, prior, costs, COMPAT)

def rmse(pred, obs):
    return float(jnp.sqrt(jnp.mean((pred - obs)**2)))

# =============================================================================
# MAIN
# =============================================================================

def compare():
    print("\nMODEL COMPARISON (RMSE)\n" + "-"*70)
    print(f"{'Model':<20} {'Breitbart':>12} {'PinkNews':>12} {'NPR':>12} {'Avg':>10}")
    print("-"*70)

    results = {}
    for ptype, priors in [("Flat", FLAT_PRIORS), ("Hier", HIER_PRIORS)]:
        for ctype, cfn in [("General", lambda o: COST_GENERAL), ("Outlet", lambda o: COST_OUTLET[o])]:
            name = f"{ptype} + {ctype}"
            rs = {o: rmse(run(priors[o], cfn(o)), OBSERVED[o]) for o in OBSERVED}
            results[name] = rs
            print(f"{name:<20} {rs['Breitbart']:>12.4f} {rs['PinkNews']:>12.4f} {rs['NPR']:>12.4f} {sum(rs.values())/3:>10.4f}")

    b = sum(results["Flat + General"].values())/3
    print(f"\nBaseline: {b:.4f}")
    print(f"  + Outlet costs:      Δ = {b - sum(results['Flat + Outlet'].values())/3:+.4f}")
    print(f"  + Hier priors:       Δ = {b - sum(results['Hier + General'].values())/3:+.4f}")
    print(f"  + Both:              Δ = {b - sum(results['Hier + Outlet'].values())/3:+.4f}")

def main():
    parser = argparse.ArgumentParser(description="RSA Model for Trans REs")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--outlet", choices=["Breitbart", "PinkNews", "NPR"])
    parser.add_argument("--prior", choices=["flat", "hier"], default="hier")
    parser.add_argument("--cost", choices=["general", "outlet"], default="outlet")
    args = parser.parse_args()

    if args.compare:
        compare()
    else:
        priors = HIER_PRIORS if args.prior == "hier" else FLAT_PRIORS
        cfn = COST_OUTLET if args.cost == "outlet" else {o: COST_GENERAL for o in OBSERVED}
        outlets = [args.outlet] if args.outlet else list(OBSERVED)

        for o in outlets:
            pred = run(priors[o], cfn[o])
            print(f"\n{o} ({args.prior}/{args.cost}):")
            for u in Utterance:
                print(f"  {UTT_NAMES[u]:<20} pred={float(pred[u]):.3f}  obs={float(OBSERVED[o][u]):.3f}")
            print(f"  RMSE: {rmse(pred, OBSERVED[o]):.4f}")

if __name__ == "__main__":
    main()
