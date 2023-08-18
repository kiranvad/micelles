# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from sasdata.dataloader.loader import Loader 
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment

from bumps.names import FitProblem, FreeVariables
from bumps.fitters import * 

import glob, os, shutil, pdb, time, datetime, json

TESTING = True 
SLD_CORE = 1.85
SLD_CORONA = 0.817
SLD_SOLVENT_LIST = {'dTHF': 6.349, 'THF': 0.183, 'D2O':6.36, 
'H2O':-0.561, 'dCF': 3.156, 'dTol':5.664, 'dAcetone':5.389,
'dTHF0':6.360, 'dTHF25':6.357, 'dTHF50':6.355, 'dTHF75':6.352,'hTHF':1.0
}
NUM_STEPS = 5 if TESTING else 5e2

SI_FILE_LOC = './EXPTDATA_V2/sample_info_OMIECS.csv'
DATA_DIR = './EXPTDATA_V2/inco_bg_sub/'

# %%
def load_data_from_file(fname, use_trim=False):
    SI = pd.read_csv(SI_FILE_LOC)
    flag = SI["Filename"]==fname
    metadata = SI[flag]
    loader = Loader()
    data = loader.load(DATA_DIR+'%s'%fname)[0]
    
    if not use_trim:
        data.qmin = min(data.x)
        data.qmax = max(data.x)
    else:
        data.qmin = min(data.x)
        data.qmax = data.x[-metadata['Highq_trim']]
        
    return data, metadata

# %%
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

    bumps_model.d_penetration.range(0.75, 1.0)    
    bumps_model.scale.range(1e-15, 1e-5)
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

def fit_files_model(fnames, model, savedir):
    start = time.time()
    datasets = []
    metadatasets = []
    for f in fnames:
        data, metadata = load_data_from_file(f, use_trim=True)
        datasets.append(data)
        metadatasets.append(metadata)

    print('Fitting the following sample : \n', metadata)
    SLD_SOLVENT = SLD_SOLVENT_LIST[metadata.Solvent.values[0]]
    sas_model, bumps_model = setup_model(model)
    bumps_model.sld_solvent.value = SLD_SOLVENT
    free = FreeVariables(names=['data_%d'%i for i in range(len(datasets))],
                        radius_core = bumps_model.radius_core,
                        radius_core_pd = bumps_model.radius_core_pd,
                        rg = bumps_model.rg,
                        )
    free.radius_core.range(20.0, 200.0)
    free.radius_core_pd.range(0.0, 0.5)
    free.rg.range(0.0, 200.0)

    print('Using the following model for fitting : \n', sas_model.info.name)
    expt = [Experiment(data=data, model=bumps_model, name='data_%d'%i) for i,data in enumerate(datasets)]
    problem = FitProblem(expt, freevars=free)
    driver = FitDriver(fitclass=DEFit, problem=problem, mapper=None, steps=NUM_STEPS)
    driver.clip() # make sure fit starts within domain
    x0 = problem.getp()
    x, fx = driver.fit()
    problem.setp(x)
    dx = driver.stderr()
    print("final chisq", problem.chisq_str())
    driver.show_err() 
    print('Final fitting parameters for files: ', fnames)
    print('Parameter Name\tFitted value')

    model_pars = problem.model_parameters()["models"][0]
    for name, param in model_pars.items():
        if not param.fixed:
            print(name, '\t' '%.2e'%param.value)

    for name, param in problem.model_parameters()["freevars"].items():
        for p in param:
            print(p.name, '\t' '%.2e'%p.value)

    for i, data in enumerate(datasets):
        fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
        fig.subplots_adjust(wspace=0.3)
        axs[0].scatter(data.x, data.y, label='True')

        # plot predicted and data curve
        min_max_mask = (data.x >= data.qmin) & (data.x <= data.qmax)
        q_mask = data.x[min_max_mask]
        axs[0].axvline(x=data.qmin, color='k')
        axs[0].axvline(x=data.qmax, color='k')   
        axs[0].plot(q_mask, problem._models[i].fitness.theory(), label='predicted', color='tab:orange')
        axs[0].set_xlabel('q')
        axs[0].set_ylabel('I(q)')
        axs[0].legend()
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')

        # plot residuals
        residuals = problem._models[i].fitness.residuals()
        axs[1].scatter(q_mask, residuals)
        axs[1].set_title('Chisq : %.2e'%problem._models[i].chisq())
        axs[1].set_xlabel('q')
        axs[1].set_ylabel('residuals')
        axs[1].set_xscale('log')
        plt.tight_layout()
        plt.savefig(savedir+'%s.png'%fnames[i])
        plt.close()

    end = time.time()
    time_str =  str(datetime.timedelta(seconds=end-start)) 
    print('Total fitting time : %s'%(time_str))

    return datasets, metadatasets, problem, bumps_model, driver


if __name__=="__main__":
    FIT_KEYS = [116,118,129,125,127,132,134,135,136,138,139,140,931,932,933,964,965,970,971] 
    BLOCK_KEYS = [('DEG', '25b'), ('DEG', '50'), ('DEG', '75'), ('PEG', '25'), ('PEG', '50')]

    SI = pd.read_csv(SI_FILE_LOC)
    counter = 0
    json_output = {}
    SI = pd.read_csv(SI_FILE_LOC)

    def get_simul_filenames(FIT_BLOCK_KEY):
        SIMUL_FILENAMES = [] 
        for key, values in SI.iterrows():
                if values['Sample'] in FIT_KEYS:
                        if (values['EG_group']==FIT_BLOCK_KEY[0] and values['Flor_block']==FIT_BLOCK_KEY[1]):
                            fname = values['Filename']
                            SIMUL_FILENAMES.append(fname)
                            print(fname,values['EG_group'], values['Flor_block'])
        
        return SIMUL_FILENAMES

    FIT_BLOCK_KEY = ('DEG', '25b') 
    fit_name = "%s_%s"%(FIT_BLOCK_KEY[0], FIT_BLOCK_KEY[1])
    SAVE_DIR = './results_simulfit/%s/'%fit_name
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    print('Saving the results to %s'%SAVE_DIR)
    SIMUL_FILENAMES = get_simul_filenames(FIT_BLOCK_KEY)
    datasets, metadatasets, problem, bumps_model, driver = fit_files_model(SIMUL_FILENAMES, 'sph', SAVE_DIR)
    all_pars = problem.model_parameters()
    model_pars = all_pars["models"][0]
    free_pars = all_pars["freevars"]
    for i, fname in enumerate(SIMUL_FILENAMES):
        # extract fitted params of an fname
        fitted_params = {}
        file_pars = model_pars.copy()
        for name, param in free_pars.items():
            file_pars[name].set(param[i].value)
        for name, param in file_pars.items():
            fitted_params[name] = param.value
        # add current fname params to json
        json_output[fname] = fitted_params
    
    with open("./results_simulfit/output.json", 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
