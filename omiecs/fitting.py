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

import argparse, glob, os, shutil, pdb, time, datetime

def load_data_from_file(fname):
    SI = pd.read_csv('./sample_info_OMIECS.csv')
    flag = SI["Filename"]==fname.split('/')[-1]
    metadata = SI[flag]
    trim = [metadata['lowq_trim'], metadata['Highq_trim']]

    with suppress(NoKnownLoaderException):
        loader = Loader()
        data = loader.load(fname)[0]

    data.qmin = data.x[trim[0]]
    data.qmax = data.x[trim[1]]
    min_max_mask = np.logical_and(data.x >= data.qmin, data.x <= data.qmax)
    data.x = data.x[min_max_mask]
    data.y = data.y[min_max_mask]
    data.dx = data.dx[min_max_mask]
    data.dy = data.dy[min_max_mask]

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

    bumps_model.radius_core.range(20.0,60.0)
    bumps_model.radius_core_pd.range(0.0, 0.3)
    bumps_model.scale.range(0.0, 1000.0)
    bumps_model.sld_core.range(0.0, 5.0)
    bumps_model.sld_corona.range(0.0, 5.0)
    # use default bounds
    bumps_model.v_core.fixed = False 
    bumps_model.v_corona.fixed = False
    bumps_model.n_aggreg.fixed = False
    # use fixed values
    bumps_model.background.fixed = True 
    bumps_model.background.value = 0.0

    return sas_model, bumps_model

def fit_file_model(fname, model, savename):
    start = time.time()
    data, metadata = load_data_from_file(fname)
    print('Fitting the following sample : \n', metadata)
    sas_model, bumps_model = setup_model(model)
    print('Using the following model for fitting : \n', sas_model.info.name)
    cutoff = 1e-3  # low precision cutoff
    expt = Experiment(data=data, model=bumps_model, cutoff=cutoff)
    problem = FitProblem(expt)
    mapper = MPIMapper.start_mapper(problem, None, cpus=0)
    result = fit(problem, method='dream', mapper=mapper, 
                samples=1e2, init = 'cov',  verbose = True
                )
    
    print('Parameter Name\tFitted value')
    for key, param in bumps_model.parameters().items():
        if not param.fixed:
            print(key, '\t', param.value)

    fig, ax = plt.subplots()
    ax.loglog(data.x, data.y, color='k', label='True')
    ax.loglog(data.x, problem.fitness.theory(), color='k', ls='--', label='predicted')
    ax.set_xlabel('q')
    ax.set_ylabel('I(q)')
    ax.legend()
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
    filelist = glob.glob("./subtracted_incoherent/*.sub")
    model = args.model
    SAVE_DIR = './fillting_%s/'%model
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    print('Saving the results to %s'%SAVE_DIR)

    for file in filelist:
        savename = SAVE_DIR+'%s.png'%(file.split('/')[-1].split('.')[0])
        fit_file_model(file, model, savename)
        break


    