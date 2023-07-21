import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sasmodels.core import load_model
from sasdata.dataloader.loader import Loader
from sasdata.data_util.loader_exceptions import NoKnownLoaderException
from contextlib import suppress
from sasmodels.bumps_model import Model, Experiment

from bumps.names import FitProblem
from bumps.fitters import fit
from bumps.mapper import MPIMapper
from bumps.fitters import * 

import argparse, glob, os, shutil, pdb, time, datetime

# Following are most likely spherical micelles with a PHFBA core and PDEGEEA corona
MONO_ASSEM = [123,114,116,125,118,122,127,129,132,134,135,136,138,139,140,148,
150,902,906,931,932,933,934,935,960,961,962,963,964,965,966,968,969,970,971]

SLD_CORE = 1.85
SLD_CORONA = 0.817
SLD_SOLVENT_LIST = {'dTHF': 6.349, 'THF': 0.183, 'D2O':6.36, 
'H2O':-0.561, 'dCF': 3.156, 'dTol':5.664, 'dAcetone':5.389,
'dTHF0':6.360, 'dTHF25':6.357, 'dTHF50':6.355, 'dTHF75':6.352,'hTHF':1.0
}

def load_data_from_file(fname):
    SI = pd.read_csv('./sample_info_OMIECS.csv')
    flag = SI["Filename"]==fname
    metadata = SI[flag]
    with suppress(NoKnownLoaderException):
        loader = Loader()
        data = loader.load(os.getcwd()+'/subtracted_incoherent/%s'%fname)[0]
    data.qmin = min(data.x)
    data.qmax = max(data.x)
    # data.qmin = data.x[metadata['lowq_trim']]
    # data.qmax = data.x[metadata['Highq_trim']]
    # min_max_mask = np.logical_and(data.x >= data.qmin, data.x <= data.qmax)
    # data.x = data.x[min_max_mask]
    # data.y = data.y[min_max_mask]
    # data.dx = data.dx[min_max_mask]
    # data.dy = data.dy[min_max_mask]

    return data, metadata

def setup_model(model):
    if model=='sph':
        sas_model = load_model("../models/spherical_micelle.py")
        bumps_model = Model(model=sas_model)

    elif model=='cyl':
        sas_model = load_model("../models/cylindrical_micelle.py")
        bumps_model = Model(model=sas_model)
        bumps_model.length_core.range(20.0,1000.0)

    elif model=='elp':
        sas_model = load_model("../models/ellipsoidal_micelle.py")
        bumps_model = Model(model=sas_model)
        bumps_model.eps.range(1.0,20.0)

    bumps_model.radius_core.range(20.0, 200.0)
    bumps_model.radius_core_pd.range(0.001, 0.5)
    bumps_model.scale.range(0.0, 1000.0)
    # use default bounds
    bumps_model.v_core.fixed = False 
    bumps_model.v_corona.fixed = False
    bumps_model.n_aggreg.fixed = True
    # use fixed values
    bumps_model.background.fixed = True 
    bumps_model.background.value = 0.0
    bumps_model.sld_core.fixed = True 
    bumps_model.sld_core.value = SLD_CORE
    bumps_model.sld_corona.fixed = True 
    bumps_model.sld_corona.value = SLD_CORONA
    bumps_model.sld_solvent.fixed = True 

    return sas_model, bumps_model

def fit_file_model(fname, model, savename):
    start = time.time()
    data, metadata = load_data_from_file(fname)
    print('Fitting the following sample : \n', metadata)
    SLD_SOLVENT = SLD_SOLVENT_LIST[metadata.Solvent.values[0]]
    sas_model, bumps_model = setup_model(model)
    bumps_model.sld_solvent.value = SLD_SOLVENT
    print('Using the following model for fitting : \n', sas_model.info.name)
    cutoff = 1e-3  # low precision cutoff
    expt = Experiment(data=data, model=bumps_model, cutoff=cutoff)
    problem = FitProblem(expt)
    # mapper = MPIMapper.start_mapper(problem, None, cpus=0)
    driver = FitDriver(fitclass=DreamFit, problem=problem, mapper=None, samples=1e3)
    driver.clip() # make sure fit starts within domain
    x0 = problem.getp()
    x, fx = driver.fit()
    problem.setp(x)
    dx = driver.stderr()
    print("final chisq", problem.chisq_str())
    driver.show_err() 
    print('Final fitting parameters for : ', fname)
    print('Parameter Name\tFitted value')
    for key, param in bumps_model.parameters().items():
        if not param.fixed:
            print(key, '\t', param.value)

    fig, axs = plt.subplots(1,2, figsize=(4*2, 4))

    # plot predicted and data curve
    axs[0].plot(data.x, problem.fitness.theory(), label='predicted', color='tab:orange')
    axs[0].errorbar(data.x, data.y, yerr=data.dy, fmt='o', ms =5, label='True', color='tab:blue')
    axs[0].set_xlabel('q')
    axs[0].set_ylabel('I(q)')
    axs[0].legend()
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    # plot residuals
    residuals = problem.fitness.residuals()
    axs[1].scatter(data.x, residuals)
    axs[1].set_title('Chisq : %.2f'%problem.chisq())
    axs[1].set_xlabel('q')
    axs[1].set_ylabel('residuals')
    axs[1].set_xscale('log')
    plt.savefig(savename)
    plt.close()

    end = time.time()
    time_str =  str(datetime.timedelta(seconds=end-start)) 
    print('Total fitting time : %s'%(time_str))

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process parameters for fitting.')
    parser.add_argument('model', metavar='m', type=str,
                        help='Specify the analytical model to be used')
    args = parser.parse_args()
    # filelist = glob.glob("./subtracted_incoherent/*.sub")
    model = args.model
    SAVE_DIR = './results_%s/'%model
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    print('Saving the results to %s'%SAVE_DIR)

    SI = pd.read_csv('./sample_info_OMIECS.csv')
    for key, values in SI.iterrows():
        if values['Sample'] in MONO_ASSEM:
            fname = values['Filename']
        fname = 'P50F50_10_dTHF50.sub'
        savename = SAVE_DIR+'%s.png'%(fname.split('.')[0])
        fit_file_model(fname, model, savename)
        break


    