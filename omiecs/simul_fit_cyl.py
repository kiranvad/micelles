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

TESTING = False 
SLD_CORE = 1.85
SLD_CORONA = 0.817
SLD_SOLVENT_LIST = {'dTHF': 6.349, 'THF': 0.183, 'D2O':6.36, 
'H2O':-0.561, 'dCF': 3.156, 'dTol':5.664, 'dAcetone':5.389,
'dTHF0':6.360, 'dTHF25':6.357, 'dTHF50':6.355, 'dTHF75':6.352,'hTHF':1.0
}

block_params = {'DEG': {'density':1.1, 'MW':188.22},
                'PEG': {'density':1.09, 'MW': 480.0},
                'F': {'density':1.418, 'MW':254.10}
                } 
DOP = {'DEG50F25' : (45, 30), # (EG, F)
       'DEG50F25b': (48, 27), 
       'DEG50F50' : (48, 52),
       'DEG50F75' : (46.25, 78.75),
       'PEG50F25' : (41.25, 33.75),
       'PEG50F50' : (50, 50)
       }
Navg = 6.02e23 
conversion = 1e24
block_vols = {}
for key, value in block_params.items():
    block_vols[key] = conversion*((value['MW']/value['density'])/Navg)

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
def setup_model():
    sas_model = load_model("../models/cylindrical_micelle.py")
    bumps_model = Model(model=sas_model)

    # use default bounds
    bumps_model.v_core.fixed = True 
    bumps_model.v_corona.fixed = True
    # bumps_model.n_aggreg.fixed = True
    # use fixed values
    bumps_model.background.fixed = True 
    bumps_model.background.value = 0.0
    bumps_model.sld_core.fixed = True 
    bumps_model.sld_core.value = SLD_CORE
    bumps_model.sld_corona.fixed = True 
    bumps_model.sld_corona.value = SLD_CORONA
    bumps_model.sld_solvent.fixed = True 

    return sas_model, bumps_model

def fit_files_model(fnames, savedir):
    start = time.time()
    datasets = []
    metadatasets = []
    for f in fnames:
        data, metadata = load_data_from_file(f, use_trim=True)
        datasets.append(data)
        metadatasets.append(metadata)

    print('Fitting the following sample : \n', metadata)
    SLD_SOLVENT = SLD_SOLVENT_LIST[metadata.Solvent.values[0]]
    sas_model, bumps_model = setup_model()
    bumps_model.sld_solvent.value = SLD_SOLVENT
    dop = DOP[metadata["Matrix"].values[0]]
    V_CORONA = dop[0]*block_vols[metadata["EG_group"].values[0]]
    V_CORE = dop[1]*block_vols["F"] 
    bumps_model.v_core.value = V_CORE 
    bumps_model.v_corona.value = V_CORONA
    free = FreeVariables(names=['data_%d'%i for i in range(len(datasets))],
                        n_aggreg = bumps_model.n_aggreg,
                        radius_core = bumps_model.radius_core,
                        radius_core_pd = bumps_model.radius_core_pd,
                        rg = bumps_model.rg,
                        scale = bumps_model.scale,
                        length_core = bumps_model.length_core,
                        )
    free.n_aggreg.range(1.0, 400.0)
    free.radius_core.range(20.0, 200.0)
    free.radius_core_pd.range(0.0, 0.5)
    free.rg.range(0.0, 200.0)
    free.scale.range(1e-15, 1e-5)
    free.length_core.range(20.0,1000.0)

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
    JOB_START_TIME = time.time()
    FIT_KEYS = [116,118,129,125,127,132,134,135,136,138,139,140,931,932,933,964,965,970,971] 
    BLOCK_KEYS = [('DEG', '25b'), ('DEG', '50'), ('DEG', '75'), ('PEG', '25'), ('PEG', '50')]
    SI = pd.read_csv(SI_FILE_LOC)
    counter = 0
    
    def get_simul_filenames(FIT_BLOCK_KEY):
        SIMUL_FILENAMES = [] 
        for key, values in SI.iterrows():
                if values['Sample'] in FIT_KEYS:
                        if (values['EG_group']==FIT_BLOCK_KEY[0] and values['Flor_block']==FIT_BLOCK_KEY[1]):
                            fname = values['Filename']
                            SIMUL_FILENAMES.append(fname)
                            print(fname,values['EG_group'], values['Flor_block'])
        
        return SIMUL_FILENAMES

    for FIT_BLOCK_KEY in BLOCK_KEYS: 
        fit_name = "%s_%s"%(FIT_BLOCK_KEY[0], FIT_BLOCK_KEY[1])
        if not TESTING:
            BASE_SAVE_DIR = './results_simulfit_cyl/'
            SAVE_DIR = BASE_SAVE_DIR+'%s/'%(fit_name)
        else:
            BASE_SAVE_DIR = './test/'
            SAVE_DIR = './test/'

        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR)
        print('Saving the results to %s'%SAVE_DIR)
        SIMUL_FILENAMES = get_simul_filenames(FIT_BLOCK_KEY)
        datasets, metadatasets, problem, bumps_model, driver = fit_files_model(SIMUL_FILENAMES, SAVE_DIR)
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
                if name=="radius_core":
                    radius_core = ((file_pars["n_aggreg"]*file_pars["v_core"])/(np.pi*file_pars["length_core"]))**(1/2)
                    fitted_params[name] = radius_core.value
                else:
                    fitted_params[name] = param.value
            # add current fname params to json
            with open(BASE_SAVE_DIR+"%s.json"%(fname.split('.')[0]), 'w', encoding='utf-8') as f:
                json.dump(fitted_params, f, ensure_ascii=False, indent=4)

        if TESTING:
            break

    JOB_END_TIME = time.time()
    time_str =  str(datetime.timedelta(seconds=JOB_END_TIME-JOB_START_TIME)) 
    print('Total fitting time : %s'%(time_str))