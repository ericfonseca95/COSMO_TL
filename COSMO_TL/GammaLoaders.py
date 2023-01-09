#!/usr/bin/env python
# coding: utf-8

import os
import gc
import time
import pickle
import webbrowser
import dask.dataframe as dd
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster, progress
from dask_ml.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR


import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from joblib import parallel_backend

import matplotlib.pyplot as plt

def gamma_convert_dataframe(txtfile):
    df = dd.read_csv(txtfile,delimiter=' ')
    df = add_chem_columns(df, txtfile).compute()
    df['x2'] = 1-df['x1']
    return df

def add_chem_columns(df, txtfile):
    rows = len(df)
    chemical_names = [txtfile.split('+')[0].split('/')[-1], txtfile.split('+')[1].split('.')[0]]
    chem1 = da.array([chemical_names[0] for i in range(rows)])
    chem2 = da.array([chemical_names[1] for i in range(rows)])
    df['Compound Name1'] = chem1
    df['Compound Name2'] = chem2
    return df


def get_profile_number(Chemical): 
    vt_csv = get_vt_df()
    index = vt_csv[vt_csv['Compound Name'] == Chemical]['Index No.'].values[0].astype(np.int32)
    return index

@dask.delayed
def get_sigma_profile(Chemical):
    chem_number = get_profile_number(Chemical)
    # rewrite using os.path.join
    # profile_dir = r'C:\Users\efons\Desktop\ML\VT_Database\profiles'
    profile_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VT_Database', 'profiles')
    files = os.listdir(profile_dir)
    for i in files:
        try:
            if int(i.split('-')[1]) == chem_number:
                profile_fname = i
        except:
            pass
    profile = np.loadtxt(profile_dir+'/'+profile_fname)[:,1]
    return profile

@dask.delayed
def get_DFT_outputs(Chemical):
    chem_number = get_profile_number(Chemical)
    # rewrite using os.path.join
    # DFT_dir = r'C:\Users\efons\Desktop\ML\VT_Database\DMOL_OUTPUT'
    DFT_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VT_Database', 'DMOL_OUTPUT')
    files = [i for i in os.listdir(DFT_dir) if 'outmol' in i]
    #DFT_fname = 'VT2005-0001-EC.outmol'
    for i in files:
        if int(i.split('-')[1]) == chem_number:
            DFT_fname = i
    DFT_outputs = []
    with open(DFT_dir+'/'+DFT_fname,'r') as file:
        lines = file.readlines()
    for idx, line in enumerate(lines):
        if line.find('DMol3/COSMO Results') > -1:
            COSMO_start = idx
        if line.find('End Computing SCF Energy/Gradient') > -1:
            COSMO_end = idx-1
    for line in lines[COSMO_start:COSMO_end]:
        if line.find('=') > -1:
            DFT_outputs.append(float(line.split('=')[-1]))
    return np.array(DFT_outputs)

@dask.delayed
def get_inputs(Chemical):
    profile = get_sigma_profile(Chemical)
    DFT_outputs = get_DFT_outputs(Chemical)
    return np.concatenate(dask.compute([profile, DFT_outputs])[0])

def get_ML_data_from_gamma_file(gamma_file):
    df1 = gamma_convert_dataframe(gamma_file)
    rows, columns = df1.shape
    input1 = get_inputs(df1['Compound Name1'][0])
    input2 = get_inputs(df1['Compound Name2'][0])
    inputs_section1 = da.hstack(dask.compute([input1, input2])[0])
    print(inputs_section1.compute())
    #inputs_section1 = np.vstack((input1, input2)).reshape(1,-1)[0]
    df2 = df1
    col_dict = {}
    for col in df2.columns:
        if '1' in col:
            col_dict[col] = col[:-1]+'2'
        if '2' in col:
            col_dict[col] = col[:-1]+'1'
    df2 = df2.rename(columns=col_dict)
    rows, columns = df2.shape
    input1 = get_inputs(df1['Compound Name1'][0])
    input2 = get_inputs(df1['Compound Name2'][0])
    inputs_section2 = da.hstack(dask.compute([input1, input2])[0])
    print(inputs_section2.compute())
    #inputs_section2 = np.vstack((input1, input2)).reshape(1,-1)[0]
    inputs=da.vstack(([inputs_section1 for i in range(rows)], [inputs_section2 for i in range(rows)])).compute()
    print(inputs)
    input_df = pd.DataFrame(inputs)
    mix_df = pd.concat((df1, df2), axis=0)
    mix_df = mix_df.reset_index(drop=True)
    mix_df = pd.concat((mix_df, input_df), axis=1)
    return mix_df

def save_ML_dataframe(file, save_dir):
    data = get_ML_data_from_gamma_file(file)
    data.to_pickle(save_dir+'/'+file.split('/')[-1].split('.')[0]+'.pickle')
    return


def gamma_convert_dataframe(txtfile):
    df = dd.read_csv(txtfile,delimiter=' ')
    df = add_chem_columns(df, txtfile).compute()
    df['x2'] = 1-df['x1']
    return df

def add_chem_columns(df, txtfile):
    rows = len(df)
    chemical_names = [txtfile.split('+')[0].split('/')[-1], txtfile.split('+')[1].split('.')[0]]
    chem1 = da.array([chemical_names[0] for i in range(rows)])
    chem2 = da.array([chemical_names[1] for i in range(rows)])
    df['Compound Name1'] = chem1
    df['Compound Name2'] = chem2
    return df

def get_vt_df():
    vt_path = os.path.join(os.path.dirname(__file__), 'data', 'VT_Database')
    vt_csv = pd.read_csv(os.path.join(vt_path,'VT-2005_Sigma_Profile_Database_Index_v2.csv'))
    return vt_csv

def get_tr_df():
    tr_path = os.path.join(os.path.dirname(__file__), 'data', 'Solvation_Database','cosmo_files')
    tr_csv = pd.read_excel(os.path.join(tr_path, 'MNSol_alldata.xls'))
    return tr_csv

def get_profile_number(Chemical):
    vt_path = os.path.join(os.path.dirname(__file__), 'data', 'VT_Database')
    vt_csv = pd.read_csv(os.path.join(vt_path,'VT-2005_Sigma_Profile_Database_Index_v2.csv'))
    index = vt_csv[vt_csv['Compound Name'] == Chemical]['Index No.'].values[0].astype(np.int32)
    return index

@dask.delayed
def get_sigma_profile(Chemical):
    chem_number = get_profile_number(Chemical)
    # profile_dir = r'C:\Users\efons\Desktop\ML\VT_Database\profiles'
    # rewrite with os.path.join
    profile_dir = os.path.join(os.path.dirname(__file__), 'data', 'VT_Database', 'profiles')
    files = os.listdir(profile_dir)
    for i in files:
        try:
            if int(i.split('-')[1]) == chem_number:
                profile_fname = i
        except:
            pass
    profile = np.loadtxt(profile_dir+'/'+profile_fname)[:,1]
    return profile

@dask.delayed
def get_DFT_outputs(Chemical):
    chem_number = get_profile_number(Chemical)
    # DFT_dir = r'C:\Users\efons\Desktop\ML\VT_Database\DMOL_OUTPUT'
    # rewrite with os.path.join
    DFT_dir = os.path.join(os.path.dirname(__file__), 'data', 'VT_Database', 'DMOL_OUTPUT')
    files = [i for i in os.listdir(DFT_dir) if 'outmol' in i]
    DFT_fname = 'VT2005-0001-EC.outmol'
    for i in files:
        if int(i.split('-')[1]) == chem_number:
            DFT_fname = i
    DFT_outputs = []
    with open(DFT_dir+'/'+DFT_fname,'r') as file:
        lines = file.readlines()
    for idx, line in enumerate(lines):
        if line.find('DMol3/COSMO Results') > -1:
            COSMO_start = idx
        if line.find('End Computing SCF Energy/Gradient') > -1:
            COSMO_end = idx-1
    for line in lines[COSMO_start:COSMO_end]:
        if line.find('=') > -1:
            DFT_outputs.append(float(line.split('=')[-1]))
    return np.array(DFT_outputs)

@dask.delayed
def get_inputs(Chemical):
    profile = get_sigma_profile(Chemical)
    DFT_outputs = get_DFT_outputs(Chemical)
    return np.concatenate(dask.compute([profile, DFT_outputs])[0])

def get_ML_data_from_gamma_file(gamma_file):
    df1 = gamma_convert_dataframe(gamma_file)
    rows, columns = df1.shape
    input1 = get_inputs(df1['Compound Name1'][0])
    input2 = get_inputs(df1['Compound Name2'][0])
    inputs_section1 = da.hstack(dask.compute([input1, input2])[0])
    print(inputs_section1.compute())
    #inputs_section1 = np.vstack((input1, input2)).reshape(1,-1)[0]
    df2 = df1
    col_dict = {}
    for col in df2.columns:
        if '1' in col:
            col_dict[col] = col[:-1]+'2'
        if '2' in col:
            col_dict[col] = col[:-1]+'1'
    df2 = df2.rename(columns=col_dict)
    rows, columns = df2.shape
    input1 = get_inputs(df1['Compound Name1'][0])
    input2 = get_inputs(df1['Compound Name2'][0])
    inputs_section2 = da.hstack(dask.compute([input1, input2])[0])
    print(inputs_section2.compute())
    #inputs_section2 = np.vstack((input1, input2)).reshape(1,-1)[0]
    inputs=da.vstack(([inputs_section1 for i in range(rows)], [inputs_section2 for i in range(rows)])).compute()
    print(inputs)
    input_df = pd.DataFrame(inputs)
    mix_df = pd.concat((df1, df2), axis=0)
    mix_df = mix_df.reset_index(drop=True)
    mix_df = pd.concat((mix_df, input_df), axis=1)
    return mix_df

def save_ML_dataframe(file, save_dir):
    data = get_ML_data_from_gamma_file(file)
    data.to_pickle(save_dir+'/'+file.split('/')[-1].split('.')[0]+'.pickle')
    return


def gamma_convert_dataframe(txtfile):
    df = pd.read_csv(txtfile,delimiter=' ')
    df = add_chem_columns(df, txtfile)
    chemical_names = [txtfile.split('+')[0].split('/')[-1], txtfile.split('+')[1].split('.')[0]]
    new_df = pd.DataFrame()
    new_df['Compound Name1'] = [chemical_names[0]]
    new_df['Compound Name2'] = [chemical_names[1]]
    new_df['ln_gamma1'] = [np.array(df['ln_gamma1'])]
    new_df['ln_gamma2'] = [np.array(df['ln_gamma2'])]
    df['x2'] = 1-df['x1']
    return new_df

def add_chem_columns(df, txtfile):
    rows = len(df)
    chemical_names = [txtfile.split('+')[0].split('/')[-1], txtfile.split('+')[1].split('.')[0]]
    chem1 = da.array([chemical_names[0] for i in range(rows)])
    chem2 = da.array([chemical_names[1] for i in range(rows)])
    df['Compound Name1'] = chem1
    df['Compound Name2'] = chem2
    return df


def get_profile_number(Chemical): 
    vt_csv = get_vt_df()
    index = vt_csv[vt_csv['Compound Name'] == Chemical]['Index No.'].values[0].astype(np.int32)
    return index

def get_sigma_profile(Chemical):
    chem_number = get_profile_number(Chemical)
    # rewrite with os.path.join
    # profile_dir = r'C:\Users\efons\Desktop\ML\VT_Database\profiles'
    profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles')
    files = os.listdir(profile_dir)
    for i in files:
        try:
            if int(i.split('-')[1]) == chem_number:
                profile_fname = i
        except:
            pass
    profile = np.loadtxt(profile_dir+'/'+profile_fname)[:,1]
    return profile

def get_DFT_outputs(Chemical):
    chem_number = get_profile_number(Chemical)
    # rewrite with os.path.join
    # DFT_dir = r'C:\Users\efons\Desktop\ML\VT_Database\DMOL_OUTPUT'
    DFT_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'VT_Database','DMOL_OUTPUT')
    files = [i for i in os.listdir(DFT_dir) if 'outmol' in i]
    DFT_fname = 'VT2005-0001-EC.outmol'
    for i in files:
        if int(i.split('-')[1]) == chem_number:
            DFT_fname = i
    DFT_outputs = []
    with open(DFT_dir+'/'+DFT_fname,'r') as file:
        lines = file.readlines()
    for idx, line in enumerate(lines):
        if line.find('DMol3/COSMO Results') > -1:
            COSMO_start = idx
        if line.find('End Computing SCF Energy/Gradient') > -1:
            COSMO_end = idx-1
    for line in lines[COSMO_start:COSMO_end]:
        if line.find('=') > -1:
            DFT_outputs.append(float(line.split('=')[-1]))
    return np.array(DFT_outputs)

def get_inputs(Chemical):
    profile = get_sigma_profile(Chemical)
    DFT_outputs = get_DFT_outputs(Chemical)
    return np.concatenate(dask.compute([profile, DFT_outputs])[0])

def get_ML_data_from_gamma_file(gamma_file):
    df1 = gamma_convert_dataframe(gamma_file)
    rows, columns = df1.shape
    input1 = get_inputs(df1['Compound Name1'][0])
    input2 = get_inputs(df1['Compound Name2'][0])
    inputs_section1 = np.concatenate((input1, input2), axis=0)
    
    df2 = df1
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    col_dict = {}
    for col in df2.columns:
        if '1' in col:
            col_dict[col] = col[:-1]+'2'
        if '2' in col:
            col_dict[col] = col[:-1]+'1'
    df2 = df2.rename(columns=col_dict)
    
    
    rows, columns = df2.shape
    input1 = get_inputs(df1['Compound Name1'][0])
    input2 = get_inputs(df1['Compound Name2'][0])
    inputs_section2 = da.hstack(dask.compute([input1, input2])[0]) 
    inputs = np.concatenate(([inputs_section1 for i in range(rows)], [inputs_section2 for i in range(rows)]))
    
    
    
    input_df = pd.DataFrame(inputs)
    X_col_names = dict(zip(input_df.columns, ['X_'+str(int(i)) for i in np.arange(len(input_df))]))
    input_df = input_df.rename(columns=X_col_names)
    mix_df = pd.concat((df1, df2), axis=0)
    mix_df = mix_df.reset_index(drop=True)
    mix_df = pd.concat((mix_df, input_df), axis=1)
    return mix_df

def save_ML_dataframe(file, save_dir):
    data = get_ML_data_from_gamma_file(file)
    data.to_pickle(save_dir+'/'+file.split('/')[-1].split('.')[0]+'.pickle')
    return data

@dask.delayed
def save_ML_files(files, save_dir):
    data = []
    for file in files:
        data.append(dask.delayed(save_ML_dataframe)(file, save_dir))
    data = dask.compute(data)[0]
    data = pd.concat(dask.compute(data)[0])
    return data


def get_IDAC_df():
    # rewrite with os.path.join
    # IDAC_dir = r'C:\Users\efons\Desktop\ML\VT_Database\IDAC'
    IDAC_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'new_data_parquet')
    ddf = dd.read_parquet(IDAC_dir)
    return ddf

def get_nist_df():
    nist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'NIST_THERMOML')
    df = pd.read_pickle(os.path.join(nist_dir, 'test_train_50.pickle'))
    df = parse_nist_df(df)
    return df

def parse_nist_df(df):
    sig_cols = ['sigma_'+str(i)+'_1' for i in range(51)]
    sig_array = np.concatenate(df['sigma_solute'].to_numpy()).reshape(-1,51)
    for i, sig in enumerate(sig_cols):
        df[sig] = sig_array[:,i]
    sig_cols = ['sigma_'+str(i)+'_2' for i in range(51)]
    sig_array = np.concatenate(df['sigma_solvent'].to_numpy()).reshape(-1,51)
    for i, sig in enumerate(sig_cols):
        df[sig] = sig_array[:,i]
    df.drop(columns=['sigma_solute','sigma_solvent'])
    df = df.rename(columns={'area_solute':'Area_1',
                          'area_solvent':'Area_2',
                          'volume_solute':'Volume_1',
                          'volume_solvent':'Volume_2'})
    return df