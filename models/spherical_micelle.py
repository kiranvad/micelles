r"""
Spherical core with Gaussian chain micelle model
"""

import numpy as np
from sasmodels.special import sas_3j1x_x, sas_sinx_x

name = "spherical_micelle"
title = "Spherical core with a Gaussian chain micelle"
description = """ See J. Appl. Cryst. (2000). 33, 637±640
      """
category = "shape:sphere"

#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [["v_core",    "Ang^3",  4000.0, [0.0, np.inf], "", "Volume of the core (single block)"],
              ["v_corona",      "Ang^3",      4000.0, [0.0, np.inf], "", "Volume of the corona (single block)"],
              ["sld_solvent",  "Ang^-2",     1.0, [0.0, np.inf], "sld", "Solvent scattering length density"],
              ["sld_core",      "1e-6/Ang^2", 2.0, [0.0, np.inf], "sld", "Core scattering length density"],
              ["sld_corona",    "1e-6/Ang^2", 2.0,  [0.0, np.inf], "sld", "Corona scattering length density"],
              ["radius_core",   "Ang",       40.0,  [0.0, np.inf], "volume", "Radius of core ( must be >> rg )"],
              ["rg",    "Ang",       10.0,  [0.0, np.inf], "volume", "Radius of gyration of chains in corona"],
              ["d_penetration", "",           1.0,  [0.0, np.inf], "", "Factor to mimic non-penetration of Gaussian chains"],
              ["n_aggreg",      "",           67.0,  [0.0, np.inf], "", "Aggregation number of the micelle"],    
              ["x_solv",      "",           0.0,  [0.0, 1.0], "", "Core solvation fraction"],        
             ]


def Iq(q,
        v_core=4000,
        v_corona=4000,
        sld_solvent=1,
        sld_core=2,
        sld_corona=1,
        radius_core=40,
        rg=10,
        d_penetration=1,
        n_aggreg=67,
        x_solv = 0.0
        ):
    n_aggreg = ((1-x_solv)*(4/3)*np.pi*(radius_core**3))/v_core
    v_total = n_aggreg*(v_core+v_corona)
    rho_solv = sld_solvent     # sld of solvent [1/A^2]
    rho_core = sld_core        # sld of core [1/A^2]
    rho_corona = sld_corona    # sld of corona [1/A^2]

    beta_core = v_core * (rho_core - rho_solv)
    beta_corona = v_corona * (rho_corona - rho_solv)
    beta_total = (beta_core+beta_corona)*n_aggreg

    # Self-correlation term of the core
    bes_core = sas_3j1x_x(q*radius_core)
    term1 = np.power(n_aggreg*beta_core*bes_core, 2)

    # Self-correlation term of the chains
    qrg2 = np.power(q*rg, 2)
    debye_chain = 2.0*(np.vectorize(np.expm1)(-qrg2)+qrg2)/(qrg2**2) 
    debye_chain[qrg2==0.0] = 1.0
    term2 = n_aggreg * beta_corona * beta_corona * debye_chain

    # Interference cross-term between core and chains
    chain_ampl = -np.vectorize(np.expm1)(-qrg2)/qrg2
    chain_ampl[qrg2==0.0] =  1.0 
    bes_corona = sas_sinx_x(q*(radius_core + (d_penetration * rg)))
    term3 = 2.0 * n_aggreg * n_aggreg * beta_core * beta_corona * bes_core * chain_ampl * bes_corona

    # Interference cross-term between chains
    term4 = n_aggreg * (n_aggreg - 1.0)* np.power(beta_corona * chain_ampl * bes_corona, 2)

    # I(q)_micelle : Sum of 4 terms computed above
    i_micelle = term1 + term2 + term3 + term4

    return i_micelle/(1.0)

Iq.vectorized = True  # Iq accepts an array of q values

def random():
    """Return a random parameter set for the model."""
    radius_core = 10**np.random.uniform(1, 3)
    rg = radius_core * 10**np.random.uniform(-2, -0.3)
    d_penetration = np.random.randn()*0.05 + 1
    n_aggreg = np.random.randint(3, 30)
    # volume of head groups is the core volume over the number of groups,
    # with a correction for packing fraction of the head groups.
    v_core = 4*np.pi/3*radius_core**3/n_aggreg * 0.68
    # Rg^2 for gaussian coil is a^2n/6 => a^2 = 6 Rg^2/n
    # a=2r => r = Rg sqrt(3/2n)
    # v = 4/3 pi r^3 n => v = 4/3 pi Rg^3 (3/2n)^(3/2) n = pi Rg^3 sqrt(6/n)
    tail_segments = np.random.randint(6, 30)
    v_corona = np.pi * rg**3 * np.sqrt(6/tail_segments)
    pars = dict(
        background=0,
        scale=1.0,
        v_core=v_core,
        v_corona=v_corona,
        radius_core=radius_core,
        rg=rg,
        d_penetration=d_penetration,
        n_aggreg=n_aggreg,
    )
    return pars