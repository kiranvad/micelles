import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment
from sasmodels.data import Data1D 
from bumps.names import FitProblem

sas_model = load_model("../models/cylindrical_micelle.py")
bumps_model = Model(model=sas_model)

q = np.logspace(-2, 0, 200)
cylinder_params = {
    'background':0.0,
    'scale':1.0,
    'v_core' : 4000.0,
    'v_corona' : 4000.0,
    'sld_solvent' : 1.0,
    'sld_core' : 2.0,
    'sld_corona' : 1.0,
    'radius_core': 40.0,
    'rg': 10.0,
    'length_core': 100.0,
    'd_penetration':1.0,
    }

# compute using call_kernel 
kernel = sas_model.make_kernel([q])
Iq_true = call_kernel(kernel, sphere_params) 

# compute using bumps : problem.fitness.theory()
resolution = 0.005 # set this as low as possible
dIq = resolution*Iq_true
dq = resolution*q
data = Data1D(q, Iq_true, dx=dq, dy=dIq) 
sas_model = load_model("../models/cylindrical_micelle.py")
bumps_model = Model(model=sas_model)
cutoff = 1e-3  # low precision cutoff
expt = Experiment(data=data, model=bumps_model, cutoff=cutoff)
problem = FitProblem(expt)
