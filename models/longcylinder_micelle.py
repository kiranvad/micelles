r"""
Cylindrical core with Gaussian chain micelle model
"""

import numpy as np
from math import expm1
from sasmodels.special import sas_2J1x_x, sas_sinx_x, sas_Si
from scipy.special import j0 as sas_J0x 

name = "wormlike_micelle"
title = "Very long cylindrical core with a Gaussian chain micelle"
description = """ See J. Appl. Cryst. (2000). 33, 637±640
      """
category = "shape:cylinder"

#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [["v_core",    "Ang^3",  4000.0, [0.0, np.inf], "", "Volume of the core (single block)"],
              ["v_corona",      "Ang^3",      4000.0, [0.0, np.inf], "", "Volume of the corona (single block)"],
              ["sld_solvent",  "Ang^-2",     1.0, [0.0, np.inf], "sld", "Solvent scattering length density"],
              ["sld_core",      "1e-6/Ang^2", 2.0, [0.0, np.inf], "sld", "Core scattering length density"],
              ["sld_corona",    "1e-6/Ang^2", 2.0,  [0.0, np.inf], "sld", "Corona scattering length density"],
              ["radius_core",   "Ang",       40.0,  [0.0, np.inf], "volume", "Radius of core ( must be >> rg )"],
              ["length_core",   "Ang",       100.0,  [0.0, np.inf], "volume", "Length of core ( must be >> rg )"],              
              ["rg",    "Ang",       10.0,  [0.0, np.inf], "volume", "Radius of gyration of chains in corona"],
              ["d_penetration", "",           1.0,  [0.0, np.inf], "", "Factor to mimic non-penetration of Gaussian chains"],
              ["n_aggreg",      "",           67.0,  [0.0, np.inf], "", "Aggregation number of the micelle"],            
             ]

def Iq(q,
       v_core=4000,
       v_corona=4000,
       sld_solvent=1,
       sld_core=2,
       sld_corona=2,
       radius_core=40,
       length_core=100,
       rg=10.0,
       d_penetration=1,
       n_aggreg=67
       ):
    n_aggreg = (np.pi * np.power(radius_core, 2.0) * length_core)/v_core
    v_total = n_aggreg*(v_core+v_corona)
    rho_solv = sld_solvent     # sld of solvent [1/A^2]
    rho_core = sld_core        # sld of core [1/A^2]
    rho_corona = sld_corona    # sld of corona [1/A^2]

    beta_core = v_core * (rho_core - rho_solv)
    beta_corona = v_corona * (rho_corona - rho_solv)

    # Self-correlation term of the core
    ff_crossec = sas_2J1x_x(q*radius_core)
    ff_thinrod_term1 = sas_Si(q*length_core)/(q*length_core)
    ff_thinrod_term2 = sas_sinx_x((q*length_core)/2)
    ff_thinrod = 2*ff_thinrod_term1 - np.power(ff_thinrod_term2, 2)

    # print(ff_crossec, ff_thinrod_term1, ff_thinrod_term2)

    Fs = np.power(ff_crossec, 2)*ff_thinrod
    term1 = np.power(n_aggreg*beta_core, 2)*Fs 

    # Self-correlation term of the chains
    qrg2 = np.power(q*rg, 2)
    debye_chain = 2.0*(np.vectorize(expm1)(-qrg2)+qrg2)/(qrg2**2) 
    debye_chain[qrg2==0.0] = 1.0
    term2 = n_aggreg * (beta_corona**2) * debye_chain

    # Interference cross-term between core and chains
    chain_ampl = -np.vectorize(expm1)(-qrg2)/qrg2
    chain_ampl[qrg2==0.0] =  1.0 
    ff_chain = sas_J0x(q*(radius_core + (d_penetration*rg)))
    Ssc = chain_ampl*ff_crossec*ff_chain*ff_thinrod
    term3 = 2.0 * (n_aggreg**2) * beta_core * beta_corona * Ssc

    # Interference cross-term between chains
    Scc = (chain_ampl**2) * (ff_chain**2) * ff_thinrod
    term4 = n_aggreg * (n_aggreg - 1.0)* (beta_corona**2)*Scc

    # I(q)_micelle : Sum of 4 terms computed above
    i_micelle = term1 + term2 + term3 + term4 
    # i_micelle = term1   
    # print('Different terms : ', term1, term2, term3, term4)

    # Normalize intensity by total volume
    return i_micelle


Iq.vectorized = True  # Iq does not accept an array of q values

def random():
    """Return a random parameter set for the model."""
    radius_core = 10**np.random.uniform(1, 3)
    rg = radius_core * 10**np.random.uniform(-2, -0.3)
    length_core = 100**np.random.uniform(1, 3)
    d_penetration = np.random.randn()*0.05 + 1
    n_aggreg = np.random.randint(3, 30)
    # volume of head groups is the core volume over the number of groups,
    # with a correction for packing fraction of the head groups.
    v_core = (np.pi*radius_core**2 * length_core)/n_aggreg
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
        length_core = length_core,
        d_penetration=d_penetration,
        n_aggreg=n_aggreg,
    )
    return pars