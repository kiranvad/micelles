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
FIT_KEYS = [116,118,129,125,127,132,134,135,136,138,139,140,931,932,933,964,965,970,971]
TESTING = False 
SLD_CORE = 1.85
SLD_CORONA = 0.817
SLD_SOLVENT_LIST = {'dTHF': 6.349, 'THF': 0.183, 'D2O':6.36, 
'H2O':-0.561, 'dCF': 3.156, 'dTol':5.664, 'dAcetone':5.389,
'dTHF0':6.360, 'dTHF25':6.357, 'dTHF50':6.355, 'dTHF75':6.352,'hTHF':1.0
}

def load_data_from_file(fname, use_trim=False):
    SI = pd.read_csv('./sample_info_OMIECS.csv')
    flag = SI["Filename"]==fname
    metadata = SI[flag]
    with suppress(NoKnownLoaderException):
        loader = Loader()
        data = loader.load(os.getcwd()+'/subtracted_incoherent/%s'%fname)[0]
    
    if not use_trim:
        data.qmin = min(data.x)
        data.qmax = max(data.x)
    else:
        data.qmin = data.x[metadata['lowq_trim']]
        data.qmax = data.x[metadata['Highq_trim']]
        
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
    bumps_model.radius_core_pd.range(0.0, 0.5)
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
    data, metadata = load_data_from_file(fname, use_trim=True)
    print('Fitting the following sample : \n', metadata)
    SLD_SOLVENT = SLD_SOLVENT_LIST[metadata.Solvent.values[0]]
    sas_model, bumps_model = setup_model(model)
    bumps_model.sld_solvent.value = SLD_SOLVENT
    print('Using the following model for fitting : \n', sas_model.info.name)
    cutoff = 1e-3  # low precision cutoff
    expt = Experiment(data=data, model=bumps_model, cutoff=cutoff)
    problem = FitProblem(expt)
    # mapper = MPIMapper.start_mapper(problem, None, cpus=0)
    driver = FitDriver(fitclass=DEFit, problem=problem, mapper=None, steps=5e2)
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
            print(key, '\t', '%.2e'%param.value)

    fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
    fig.subplots_adjust(wspace=0.3)
    axs[0].errorbar(data.x, data.y, yerr=data.dy, fmt='o', 
    ms=4, label='True', markerfacecolor='none', markeredgecolor='tab:blue')
    # plot predicted and data curve
    min_max_mask = (data.x >= data.qmin) & (data.x <= data.qmax)
    q_mask = data.x[min_max_mask]
    axs[0].axvline(x=data.qmin, color='k')
    axs[0].axvline(x=data.qmax, color='k')   
    # masking from sasmodels : sasmodels/direct_model.py#L270
    axs[0].plot(q_mask, problem.fitness.theory(), label='predicted', color='tab:orange')
    # the fitness computes the kernel on set of data.x computed using resolution
    # see sasmodels/direct_model.py#L259
    # more precisely it is set using resolution.Pinhole1D see /sasmodels/direct_model.py#L268
    axs[0].set_xlabel('q')
    axs[0].set_ylabel('I(q)')
    axs[0].legend()
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    # plot residuals
    residuals = problem.fitness.residuals()
    axs[1].scatter(q_mask, residuals)
    axs[1].set_title('Chisq : %.2e'%problem.chisq())
    axs[1].set_xlabel('q')
    axs[1].set_ylabel('residuals')
    axs[1].set_xscale('log')
    plt.tight_layout()
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
    if not TESTING:
        SAVE_DIR = './results_%s/'%model
    else:
        SAVE_DIR = './test/'
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    print('Saving the results to %s'%SAVE_DIR)

    SI = pd.read_csv('./sample_info_OMIECS.csv')
    for key, values in SI.iterrows():
        if values['Sample'] in FIT_KEYS:
            fname = values['Filename']
            if TESTING:
                fname = 'P50F50_10_dTHF50.sub'
            savename = SAVE_DIR+'%s.png'%(fname.split('.')[0])
            fit_file_model(fname, model, savename)
            if TESTING:
                break


    