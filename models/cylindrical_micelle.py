r"""
Cylindrical core with Gaussian chain micelle model
"""

import numpy as np
from math import expm1
from scipy.special import j0 as B0
from scipy.special import j1 as B1

def sas_2j1x_x(x):
    if np.isclose(x, 0.0):
        return 2.0
    else:
        return (2*B1(x))/x

def sas_sinx_x(x):
    if np.isclose(x, 0.0):
        return 1.0
    else:
        return np.sin(x)/x
    
def sas_j0x(x):
    if np.isclose(x, 0.0):
        return 1.0
    else:
        return B0(x)  
    
def orientational_average(f, num_alpha = 200):
    """ Compute orientational average

    f should be a function of alpha
    """
    alpha = np.linspace(0, np.pi, num=num_alpha)
    integrand = [f(a)*np.sin(a) for a in alpha]

    return np.trapz(integrand, x = alpha)

def psi(q, R, L, a):
    x = q*R*np.sin(a)
    y = 0.5*q*L*np.cos(a)

    t1 = sas_2j1x_x(x)
    t2 = sas_sinx_x(y)

    return t1*t2

def sigma(q, R, L, a):
    x = q*R*np.sin(a)
    y = 0.5*q*L*np.cos(a)

    t1 = (R/(R+L))*(sas_2j1x_x(x))*np.cos(y)
    t2 = (L/(R+L))*(sas_j0x(x))*(sas_sinx_x(y))

    return t1+t2


name = "cylindrical_micelle"
title = "Cylindrical core with a Gaussian chain micelle"
description = """ See J. Appl. Cryst. (2000). 33, 637±640
      """
category = "shape:cylinder"

#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [["v_core",    "Ang^3",  4000.0, [0.0, np.inf], "", "Volume of the core (single block)"],
              ["v_corona",      "Ang^3",      4000.0, [0.0, np.inf], "", "Volume of the corona (single block)"],
              ["sld_solvent",  "Ang^-2",     1.0, [0.0, np.inf], "sld", "Solvent scattering length density"],
              ["sld_core",      "1e-6/Ang^2", 2.0, [0.0, np.inf], "sld", "Core scattering length density"],
              ["sld_corona",    "1e-6/Ang^2", 2.0,  [0.0, np.inf], "sld", "Corona scattering length density"],
              ["radius_core",   "Ang",       40.0,  [0.0, np.inf], "", "Radius of core ( must be >> rg )"],
              ["length_core",   "Ang",       100.0,  [0.0, np.inf], "", "Length of core ( must be >> rg )"],              
              ["rg",    "Ang",       10.0,  [0.0, np.inf], "", "Radius of gyration of chains in corona"],
              ["d_penetration", "",           1.0,  [-np.inf, np.inf], "", "Factor to mimic non-penetration of Gaussian chains"],
              ["n_aggreg",      "",           67.0,  [-np.inf, np.inf], "", "Aggregation number of the micelle"],            
             ]

def Iq(q,
       v_core=4000,
       v_corona=4000,
       sld_solvent=1,
       sld_core=2,
       sld_corona=2,
       radius_core=40,
       rg=10,
       length_core=100,
       d_penetration=1,
       n_aggreg=67
       ):
    
    v_total = n_aggreg*(v_core+v_corona)
    rho_solv = sld_solvent     # sld of solvent [1/A^2]
    rho_core = sld_core        # sld of core [1/A^2]
    rho_corona = sld_corona    # sld of corona [1/A^2]

    beta_core = v_core * (rho_core - rho_solv)
    beta_corona = v_corona * (rho_corona - rho_solv)

    # Self-correlation term of the core
    bes_core = lambda a : psi(q, radius_core, length_core, a)
    Fs = orientational_average(lambda a : bes_core(a)**2)
    term1 = np.power(n_aggreg*beta_core, 2)*Fs

    # Self-correlation term of the chains
    qrg2 = np.power(q*rg, 2)
    debye_chain = 1.0 if qrg2==0.0 else 2.0*(expm1(-qrg2)+qrg2)/(qrg2**2)
    term2 = n_aggreg * (beta_corona**2) * debye_chain

    # Interference cross-term between core and chains
    qrg = q*rg
    chain_ampl =  1.0 if qrg==0.0 else -expm1(-qrg)/qrg
    bes_corona = lambda a : sigma(q, 
                                  radius_core+ d_penetration*rg, 
                                  length_core+ 2*d_penetration*rg,
                                  a
                                  )
    Ssc = chain_ampl*orientational_average(lambda a : bes_core(a)*bes_corona(a))
    term3 = 2.0 * (n_aggreg**2) * beta_core * beta_corona * Ssc

    # Interference cross-term between chains
    Scc = (chain_ampl**2)*orientational_average(lambda a : bes_corona(a)**2)
    term4 = n_aggreg * (n_aggreg - 1.0)* (beta_corona**2)*Scc

    # I(q)_micelle : Sum of 4 terms computed above
    i_micelle = term1 + term2 + term3 + term4

    # Normalize intensity by total volume
    return i_micelle/v_total

Iq.vectorized = False  # Iq does not accept an array of q values

def random():
    """Return a random parameter set for the model."""
    radius_core = 10**np.random.uniform(1, 3)
    rg = radius_core * 10**np.random.uniform(-2, -0.3)
    length_core = 100**np.random.uniform(1, 3)
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
        length_core = length_core,
        d_penetration=d_penetration,
        n_aggreg=n_aggreg,
    )
    return pars