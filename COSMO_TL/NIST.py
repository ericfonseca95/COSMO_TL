import pandas as pd
import numpy as np
import json
import os
from . import GammaLoaders as gl
from difflib import get_close_matches, SequenceMatcher
import matplotlib.pyplot as plt
from . import sigma_functions as sf

class ThermoML():
    def __init__(self, doi):
        # use the doi to get the json file from the data folder of the package
        with open(os.path.join(os.path.dirname(__file__), 'data', 'NIST_THERMOML', doi+'.json'),'r') as file:
            j = json.load(file)
        ###with open('/Users/efons/Desktop/'+doi+'.json','r') as file:
        ###    j = json.load(file)
        self.json = j
        #self.reg_df = get_reg_df

        # Get DataFrame keys for reagents
        num = []
        sample = []
        data = self.json
        for i in data['Compound']:
            num.append(i['RegNum']['nOrgNum'])
            sample.append(i['sCommonName'][0])
        reg_df= pd.DataFrame()
        reg_df['RegNum'] = num
        reg_df['Compound'] = sample

        # Get DataFrame keys for Compounds
        num = []
        sample = []
        for i in data['Compound']:
            num.append(i['RegNum']['nOrgNum'])
            sample.append(i['sCommonName'][0])
        reg_df= pd.DataFrame()
        reg_df['RegNum'] = num
        reg_df['Compound'] = sample
        self.reg_df = reg_df

        # GET ALL THE DATAPOINTS
        dfs = []
        k2 = []
        self.var_values = []
        self.prop_values = []
        self.comp = []
        MixData = [PureOrMixtureData(i) for i in data['PureOrMixtureData']]
        self.dfs = [i.df for i in MixData]
        #print(dfs)
        self.data = data
        self.MixData = MixData
        self.df = pd.concat(self.dfs).reset_index(drop=True).dropna()
        self.df = pd.merge(self.df, self.reg_df, left_on='RegNum1', right_on='RegNum', copy=False)
        self.df['Compound Name 1'] = self.df['Compound']
        self.df = self.df.drop(columns='Compound')
        self.df = pd.merge(self.df, self.reg_df, left_on='RegNum2', right_on='RegNum', copy=False)
        self.df['Compound Name 2'] = self.df['Compound']
        self.df = self.df.drop(columns=['Compound', 'RegNum_x','RegNum_y'])
        return
    
class PureOrMixtureData():
    def __init__(self, json_data):
        dfs = []
        k2 = []
        self.var_values = []
        self.prop_values = []
        self.comp = []
        compound = json_data['Component']
        compounds = []
        self.data = json_data
        for c in compound:
            compounds.append(c['RegNum']['nOrgNum'])
        self.comp.append(compounds)

        df = pd.DataFrame()
        for j in json_data['Variable']:
            var_name = j['VariableID']['VariableType']['tml_elements'][0]
        prop_name = 'prop_name'

        prop_names = []
        prop_number = []
        k2 = []
        for j in json_data['Property']:
            og = j
            j = j['Property-MethodID']['PropertyGroup']
            for prop in j['tml_elements']:
                for k in j[prop].keys():
                    if k == 'ePropName':
                        prop_names.append(str(j[prop][k]))
                        prop_number.append(og['nPropNumber'])
        self.prop_names = prop_names[0]
        self.prop_number = prop_number[0]


        self.var_names = []
        self.var_number = []
        #print('VARIABLES : ', len(json_data['Variable']))
        for j in json_data['Variable']:
            for k in j['VariableID']:
                keys = j['VariableID']['VariableType']['tml_elements']
                for key in keys:
                    self.var_names.append(j['VariableID']['VariableType'][key])
                    self.var_number.append(j['nVarNumber'])
        self.var_names1 = self.var_names
        self.var_names = np.unique(self.var_names)
        
        for index, j in enumerate(json_data['NumValues']):
            #for var in j['VariableValue']:
            self.var_values.append(j['VariableValue'][0]['nVarValue'])
            self.prop_values.append(j['PropertyValue'][0]['nPropValue'])
            compounds = np.zeros((len(self.var_values), 2))
            for index in range(len(self.var_values)):
                compounds[index,:] = self.comp[-1]
            self.compounds = np.array(compounds)
        rows = []
        for index1, j in enumerate(json_data['NumValues']):
            row = np.array([np.nan for i in range(len(j['VariableValue']))])
            for index2, var in enumerate(j['VariableValue']):
                row[index2] = var['nVarValue']
                var_num = var['nVarNumber']
            rows.append(row)
        array = np.concatenate([rows]).reshape(index1+1, index2+1)
        self.array = array
        self.df = pd.DataFrame()
        for index, var in enumerate(self.var_names):
            self.df[var] = array[:, index]
        self.df['Variable'] = self.var_values
        self.df['Property'] = self.prop_values
        self.df['RegNum1'] = self.compounds[:,0]
        self.df['RegNum2'] = self.compounds[:,1]
        self.variables = json_data['Variable']
        self.properties = json_data['Property']
        return
    
class Variable():
    def __init__(self, json_data):
        data = json_data['Variable']
        self.data = data
        var_num = []
        var_name = []
        for i in data:
            var_num.append(i['nVarNumber'])
            v = list(i['VariableID']['VariableType']['tml_elements'])
            self.v = v
            var_name.append(i['VariableID']['VariableType'][v[0]])
        self.var_num = var_num
        self.var_name = var_name
        self.df = pd.DataFrame()
        self.df['Variable'] = var_name
        self.df['nVarNumber'] = var_num
        return

class Property():
    def __init__(self, json_data):
        data = json_data['Property']
        self.data = data
        prop_num = []
        prop_name = []
        for i in data:
            prop_num.append(i['nPropNumber'])
            prop_groups = i['Property-MethodID']['PropertyGroup']['tml_elements']
            for group in prop_groups:
                prop_name.append(i['Property-MethodID']['PropertyGroup'][group]['ePropName'])
        self.prop_num = prop_num
        self.prop_name = prop_name
        self.df = pd.DataFrame()
        self.df['Property'] = prop_name
        self.df['nPropNumber'] = prop_num
        return
    
class NumValues():
    def __init__(self, json_data):
        data = json_data['NumValues']
        self.data = data
        
        var_num = []
        var_values = []
        for index1, j in enumerate(data):
            row = np.array([np.nan for i in range(len(j['VariableValue']))])
            for index2, var in enumerate(j['VariableValue']):
                row[index2] = var['nVarValue']
                var_num.append(var['nVarNumber'])
            var_values.append(row)
        self.var_num = var_num
        self.var_values = var_values
        
        print(len(self.var_num))
        prop_num = []
        prop_values = []
        for index1, j in enumerate(data):
            prop_value = np.array([np.nan for i in range(len(j['PropertyValue']))])
            for index2, var in enumerate(j['PropertyValue']):
                prop_value[index2] = var['nPropValue']
                prop_num.append(var['nPropNumber'])
            prop_values.append(prop_value)
        
        self.prop_rows = prop_values
        self.prop_num = prop_num
        n_cols = len(np.unique(var_num))
        indicies = [np.where(var_num==i)[0] for i in np.unique(var_num)]
        self.indicies = indicies
        print(self.indicies)
        rows = np.concatenate(prop_values).reshape(-1)
        self.rows = rows
        new = []
        print(n_cols)
        for i in range(n_cols):
            new.append(rows[indicies[i]].reshape(-1, 1))
        self.new = new
        self.new = np.hstack(new).reshape(-1, n_cols)
        self.df = pd.DataFrame(self.new)
        self.df['nPropNumber'] = self.prop_num
        self.df['nVarNumber'] = self.var_num
        self.df['nVarValue'] = self.var_values
        og_cols = np.arange(self.new.shape[1])
        new_cols = ['nVarNumber_' + i for i in np.arange(1, self.new.shape[1]+1).astype(np.str_)]
        cols = dict(zip(og_cols, new_cols))
        print(cols)
        self.df = self.df.rename(columns=cols)
        self.nums = var_num
        return

class NumValues():
    def __init__(self, json_data):
        data = json_data['NumValues']
        self.data = data
        
        var_num = []
        var_values = []
        for index1, j in enumerate(data):
            row = np.array([np.nan for i in range(len(j['VariableValue']))])
            for index2, var in enumerate(j['VariableValue']):
                row[index2] = var['nVarValue']
                var_num.append(var['nVarNumber'])
            var_values.append(row)
        self.var_num = var_num
        self.var_values = var_values
        prop_num = []
        prop_values = []
        for index1, j in enumerate(data):
            prop_value = np.array([np.nan for i in range(len(j['PropertyValue']))])
            for index2, var in enumerate(j['PropertyValue']):
                prop_value[index2] = var['nPropValue']
                prop_num.append(var['nPropNumber'])
            prop_values.append(prop_value)
        self.prop_num = prop_num
        self.prop_values = prop_values
        
        var_cols = ['nVarNumber_'+str(i) for i in np.unique(self.var_num)]
        prop_cols = ['nPropNumber_'+str(i) for i in np.unique(self.prop_num)]
        
        var_df = pd.DataFrame(var_values)
        prop_df = pd.DataFrame(prop_values)
        var_df = var_df.rename(columns=dict(zip(var_df.columns, var_cols)))
        prop_df = prop_df.rename(columns=dict(zip(prop_df.columns, prop_cols)))
        self.var_df = var_df
        self.prop_df = prop_df
        self.df = pd.concat([self.var_df, self.prop_df], axis=1)
        return

class PureOrMixtureData():
    def __init__(self, json_data):
        dfs = []
        k2 = []
        self.var_values = []
        self.prop_values = []
        self.comp = []
        compound = json_data['Component']
        compounds = []
        self.data = json_data
        for c in compound:
            compounds.append(c['RegNum']['nOrgNum'])
        self.comp.append(compounds)

        self.numValues = NumValues(json_data)
        self.prop = Property(json_data)
        self.prop_num = self.prop.prop_num
        self.prop_name = self.prop.prop_name
        prop_cols = dict(zip(['nPropNumber_'+str(i) for i in self.prop_num], self.prop_name))
        self.prop_cols = prop_cols
        self.variable = Variable(json_data)
        self.var_num = self.variable.var_num
        self.var_name = self.variable.var_name
        var_cols = dict(zip(['nVarNumber_'+str(i) for i in self.var_num], self.var_name))
        self.var_cols = var_cols
        self.df = self.numValues.df
        self.df = self.df.rename(columns=var_cols)
        self.df = self.df.rename(columns=prop_cols)
        self.compounds = (np.ones((len(self.df),2))*self.comp).astype(np.int32)
        self.df['RegNum1'] = self.compounds[:,0]
        self.df['RegNum2'] = self.compounds[:,1]
        return
    

class ThermoML2():
    def __init__(self, doi):
        # get data/doi.json path from the package
        with open(os.path.join(os.path.dirname(__file__), 'data', 'NIST_THERMOML', doi+'.json'),'r') as file:
            j = json.load(file)
        self.json = j
        self.df = pd.DataFrame()
        #self.reg_df = get_reg_df
        try:
            # Get DataFrame keys for reagents
            num = []
            sample = []
            data = self.json
            for i in data['Compound']:
                num.append(i['RegNum']['nOrgNum'])
                sample.append(i['sCommonName'][0])
            reg_df= pd.DataFrame()
            reg_df['RegNum'] = num
            reg_df['Compound'] = sample

            # Get DataFrame keys for Compounds
            num = []
            sample = []
            for i in data['Compound']:
                num.append(i['RegNum']['nOrgNum'])
                sample.append(i['sCommonName'][0])
            reg_df= pd.DataFrame()
            reg_df['RegNum'] = num
            reg_df['Compound'] = sample
            self.reg_df = reg_df
            self.MixData = [PureOrMixtureData(i) for i in data['PureOrMixtureData']]
            self.df = pd.concat([i.df for i in self.MixData]).reset_index(drop=True)
            self.df = pd.merge(self.reg_df, self.df, left_on='RegNum', right_on='RegNum1')
            self.df = self.df.rename(columns={'Compound':'Compound Name 1'})
            self.df = self.df.drop(columns='RegNum')
            self.df = pd.merge(self.reg_df, self.df, left_on='RegNum', right_on='RegNum2')
            self.df = self.df.rename(columns={'Compound':'Compound Name 2'})
            self.df = self.df.drop(columns='RegNum')
        except:
            pass
        return


def get_tr_sigma_profile(name):
    # define the tr_path as the data path of the package
    tr_path = os.path.join(os.path.dirname(__file__), 'data', 'Solvation_Database','cosmo_files')
    tr_csv = get_tr_df()
    tr_cosmo_fnames = [os.path.join(tr_path,i) for i in os.listdir(tr_path) if '.cosmo' in i]
    tr_fnames = [i.split('/')[-1].split('_')[0] for i in tr_cosmo_fnames]
    len(np.unique(tr_csv['Solvent'])), len(np.unique(tr_csv['SoluteName'])), tr_cosmo_fnames
    name = name.lower()
    row = tr_csv[tr_csv['SoluteName'] == name]
    lst = list(row['FileHandle'])
    chosen_fname = max(set(lst), key=lst.count)
    sigma_profile = sf.sigma_function([i for i in tr_cosmo_fnames if chosen_fname in i][0])[1]
    size = sf.get_size([i for i in tr_cosmo_fnames if chosen_fname in i][0])
    return sigma_profile, size

def get_custom_sigma(name):
    # make name prefix ../data/custom_profiles/ from the package folder "data"
    # make name suffix .cosmo
    # return sigma_profile, size
    name = name.lower()
    name = name + '.cosmo'
    directory = os.path.dirname(os.path.abspath(__file__))
    # join with the data path
    path = os.path.join(directory, 'data', 'Solvation_Database','custom_profiles')
    fname = path
    sigma_profile = sf.sigma_function(os.path.join(path, name))[1]
    size = sf.get_size(name)
    return sigma_profile, size

def similiar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_tr_df():
    tr_path = os.path.join(os.path.dirname(__file__), 'data', 'Solvation_Database','cosmo_files')
    tr_csv = pd.read_excel(os.path.join(tr_path, 'MNSol_alldata.xls'))
    return tr_csv
