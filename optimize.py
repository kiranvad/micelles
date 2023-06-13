""" Optimize custom form factors with bumps

This function describes an example of optimizing a model using bumps.

To run, use the bumps command line interface:
1. You first need to locally clone sasview and sasmodels for this to work.

git clone https://github.com/sasview/sasview.git
git clone https://github.com/sasview/sasmodels.git

2. Next, add the freshly cloned repos to your PYTHONPATH.

(on a Mac/linux) export PYTHONPATH="<your_directory>/sasview/src"

3. Run the bumps command with in the optimize.py directory
$ bumps optimize.py --preview

This should generate a plot of the sample data we are optimizing for and when you close it,
it will continue the optimization using bumps and generate the fitting plot
"""

import numpy as np
import matplotlib.pyplot as plt
from sasmodels.core import load_model
from sasmodels.direct_model import call_kernel
from sasmodels.bumps_model import Model, Experiment
from sasmodels.data import Data1D 
from bumps.names import *
from bumps.fitters import fit

# Spherical micelle in sasmodels to create a sample data
model = load_model("./models/spherical_micelle.py")
q = np.logspace(-2, 0, 200)
kernel = model.make_kernel([q])
sphere_params = {'v_core' : 4000.0,
         'v_corona' : 4000.0,
         'sld_solvent' : 1.0,
         'sld_core' : 2.0,
         'sld_corona' : 1.0,
         'radius_core': 40.0,
         'rg': 10.0,
         'd_penetration':1.0,
         'n_aggreg' : 67.0,
         }
Iq = call_kernel(kernel, sphere_params)
fig, ax = plt.subplots()
ax.loglog(q, Iq)
ax.set_ylim([1e0, 1e6])
plt.show()

# setup bumps interface for optimization
resolution = 0.005
dIq = resolution*Iq
dq = resolution*q
data = Data1D(q, Iq, dx=dq, dy=dIq)
model = Model(model=model, background=0.0)
model.radius_core.range(2.0,100.0)
cutoff = 1e-3  # low precision cutoff
M = Experiment(data=data, model=model, cutoff=cutoff)
problem = FitProblem(M)
result=fit(problem, method='dream',samples=1e2, init = 'cov',  verbose = True)