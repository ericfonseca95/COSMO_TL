import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from scipy.interpolate import interp1d
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
import pandas as pd
from . import GammaLoaders as gl


def sigma_function(sigma_file):
    data = get_sigma_data(sigma_file)
    segments = len(data)
    chg_den = np.linspace(-2.5e-2, 2.5e-2, 51)
    
    #aeff = 2.10025 # Angs
    #aeff = 0
    r_eff = 0.81764 # Angs
    aeff = np.pi*r_eff**2
    an = data[:,6] # Ang**2 normalized Area from Klamt
    rn = np.sqrt(an/np.pi) # rn per segment
    xyz = data[:,2:5]*0.5192 # Au to Ang
    x = xyz[:,0] # Ang
    y = xyz[:,1] # Ang
    z = xyz[:,2] # Ang
    d = pairwise_distances(xyz) # distance between each segment in Ang
    atom_number = data[:,1]
    sigma_cosmo = data[:,7] # e/Ang**2
    sigma_m = np.zeros(segments)
    for i in range(segments):
        numer_sum = 0
        denom_sum = 0
        for j in range(segments):
            r = rn[j]
            r_term = r**2*r_eff**2/(r**2+r_eff**2) 
            exp_term = np.exp(-d[i,j]**2/(r**2+r_eff**2))
            sum_term = r_term * exp_term
            numer = sigma_cosmo[j] * sum_term
            denom = sum_term
            numer_sum = numer_sum + numer
            denom_sum = denom_sum + denom
        sigma_m[i] = numer_sum/denom_sum
        
    dig = np.digitize(sigma_m, chg_den)
    p = np.zeros(len(chg_den))
    A_i = np.array([an[np.where(dig==i)] for i in range(len(chg_den))])/aeff
    n_i = A_i/aeff
    for i in range(len(chg_den)):
        p[i] = np.sum(an[np.where(dig==i)])
    return chg_den, p
   
def generate_sigma_profiles(ngauss=5):    
    chg_den = np.linspace(-2.5e-2, 2.5e-2, 51)
    range_den = np.ptp(chg_den)
    main_gauss = np.random.rand(1)*20*np.exp(-((chg_den)/(0.001))**2)
    mu = np.random.rand(ngauss)*range_den + min(chg_den)
    sigmas = np.random.rand(ngauss)*range_den/20
    scales = np.random.rand(ngauss)*50
    y1 = np.transpose(np.array([scales[i]*np.exp(-((chg_den-mu[i])/(sigmas[i]))**2) \
                                for i in range(ngauss)]))
    y1_sum = np.sum(y1, axis=1)
    mu2 = np.random.rand(2)*(max(chg_den)-min(chg_den))+min(chg_den)
    sigmas2 = np.random.rand(2)*range_den/2
    scales2 = np.random.rand(2)*np.max(y1_sum)
    y2 = np.transpose(np.array([scales2[i]*np.exp(-((chg_den-mu2[i])/(sigmas2[i]))**2) \
                               for i in range(2)]))
    y2_sum = np.sum(y2, axis=1)
    y1_sum = y1_sum-y2_sum
    y1_sum = y1_sum + main_gauss
    y1_final = []
    for i in y1_sum:
        if i > 0:
            y1_final.append(i)
        else:
            y1_final.append(0)
    y1_final = np.array(y1_final)
    area = np.sum(y1_final)
    return y1_final, area

def get_radii(sigma_file):
    with open(sigma_file, 'r') as sf:
        lines = sf.readlines()
        for idx, line in enumerate(lines):
            if 'coord_rad' in line:
                start = int(idx)
            if 'coord_car' in line:
                end = int(idx)
                break
        data = [i.split() for i in lines[start+2:end]]
        elem_numbers = [int(i[0]) for i in data]
        print(elem_numbers)
        radii = [float(i[-4]) for i in data]
        print(radii)
        cosmo_dict = dict(zip(elem_numbers, radii))
    return cosmo_dict

def get_sigma_data(sigma_file):
    with open(sigma_file,'r') as sf:
        lines = sf.readlines()
        for idx, line in enumerate(lines):
            if '(X, Y, Z)' in line:
                skip_index = idx
                break

    data = np.loadtxt(sigma_file, skiprows=skip_index)
    return data

def get_size(sigma_file):
    with open(sigma_file, 'r') as sf:
        lines = sf.readlines()
        for idx, line in enumerate(lines):
            if 'cosmo_data' in line:
                start = int(idx)
            if 'coord_rad' in line:
                end = int(idx)
                break
        data = [i.split('=') for i in lines[start+2:end]]
        Area = float(data[-2][-1])*0.52917**2
        Volume = float(data[-1][-1])*0.52917**3
    return Area, Volume

def Antoine(A, B, C, T):
    Psat = 10**(A - (B/(T+C)))
    return Psat


# Lets turn COSMO_GAMMA into a class?
class COSMO():
    def __init__(self, c1, c2, v1, v2, x1 = np.linspace(1e-10,0.9999999,1000), T=298.15):
        self.T = T
        self.x1 = x1
        self.c1 = np.array([i for i in c1]).reshape(1,-1)
        self.c2 = np.array([i for i in c2]).reshape(1,-1)
        
        
        self.c = np.zeros((len(c1), 2))
        self.c[:,0] = self.c1
        self.c[:,1] = self.c2
        
        #### CONSTANTS ####
        self.thresh = 0.0001
        self.compseg = 51
        self.e0 = 2.395e-4 # permitivity of free space (e**2*mole/Kcal/Ang)
        self.a_eff_prime = 7.5 #2.10025 # effective surface area (Ang**2)
        self.rgas = 0.001987 #kcal/mol/K ideal gas
        self.vnorm = 66.69 # ang^3 volume normalization constant
        self.anorm = 79.53 # ang^2 area normalization constant
        self.EPS = 3.667 # 
        self.coord = 10.0 # coordination number
        self.sigma_HB = 0.0084 #hydrogen bonding cutoff value
        self.CHB = 85580.0 # hydrogen bonding coefficient (Kcal/mol/Ang**2)
        self.sigma_1 = np.linspace(-2.5e-2, 2.5e-2, 51)
        self.sigma_2 = np.linspace(-2.5e-2, 2.5e-2, 51)
        #self.alpha_prime = 9034.97
        self.alpha_prime = 16466.72
        self.sigma_hb = 0.0084 #e/Ang2
        self.vcosmo = np.array([0, 0]).astype(np.float32)
        self.acosmo = np.sum(self.c, axis=0)
        self.vcosmo = np.array([v1, v2])
        return 
    
    def get_gamma(self):
        x1 = self.x1
        compseg = self.compseg
        ps = self.get_solution_profile()
        exchange_energy = self.get_exchange_energy()
        seg_gammas = self.get_i_activity_segments()
        seg_gamma_pr = self.get_s_activity_segments()
        ln_gamma_sg1, ln_gamma_sg2 = self.get_lngamma_SG()
        gammas1 = np.zeros(len(x1))
        gammas2 = np.zeros(len(x1))
        
        self.ln_gamma1_resid = self.acosmo[0]/self.a_eff_prime*np.sum((self.c[:,0]/self.acosmo[0])* \
                   (np.array([-np.log(self.seg_gamma_pr[:,0])+np.log(seg_gammas[:,i]) for i in range(len(self.x1))])), axis=1)
        
        self.ln_gamma2_resid = self.acosmo[1]/self.a_eff_prime*np.sum((self.c[:,1]/self.acosmo[1])*\
                   (np.array([-np.log(self.seg_gamma_pr[:,1])+np.log(seg_gammas[:,i]) for i in range(len(self.x1))])), axis=1)
        
        self.ln_gamma1 = self.ln_gamma1_resid + self.ln_gamma_sg1
        self.ln_gamma2 = self.ln_gamma2_resid + self.ln_gamma_sg2
        return self.ln_gamma1, self.ln_gamma2
        
        
    def get_solution_profile(self):
        # FIND THE MIXTURE PROFILE
        compseg = self.compseg
        ps = np.zeros((compseg, len(self.x1)))
        for i, x in enumerate(self.x1):
            ps[:,i] =(x*self.c[:,0] + (1-x)*self.c[:,1]) / \
                        np.sum(x*self.acosmo[0] + (1-x)*self.acosmo[1])
        self.ps = ps
        return ps
    
    def get_exchange_energy(self):
        sigma_tabulated = np.linspace(-0.025, 0.025, 51)
        sigma_m = np.tile(sigma_tabulated,(len(sigma_tabulated),1))
        sigma_n = np.tile(np.array(sigma_tabulated,ndmin=2).T,(1,len(sigma_tabulated)))
        sigma_acc = np.tril(sigma_n) + np.triu(sigma_m,1)
        sigma_don = np.tril(sigma_m) + np.triu(sigma_n,1)
        DELTAW = (self.alpha_prime/2)*(sigma_m + sigma_n)**2 + \
                self.CHB*np.maximum(0, sigma_acc - self.sigma_hb)*np.minimum(0, sigma_don+self.sigma_hb)
        self.exchange_energy = DELTAW
        return self.exchange_energy
    
    def get_i_activity_segments(self):
        # solvent segment activity coefficient
        x1 = self.x1
        compseg = self.compseg
        seg_gammas = np.ones((compseg, len(x1)))
        for x in range(len(self.x1)):
            counter = 0
            converge = np.ones((compseg, 1))
            seg_gamma = np.ones((compseg))
            seg_gamma_old = seg_gamma
            AA = np.exp(-self.exchange_energy /self.rgas/self.T)*self.ps[:,x]
            while np.max(converge) > self.thresh:
                counter += 1
                seg_gamma_old = seg_gamma
                seg_gamma = np.exp(-np.log(np.sum(AA*seg_gamma, axis=1)))
                seg_gamma = (seg_gamma + seg_gamma_old)/2
                converge = np.abs((seg_gamma-seg_gamma_old)/seg_gamma_old)
                if counter > 1e3 or np.max(converge) < self.thresh:
                    break
            seg_gammas[:,x] = seg_gamma
        self.seg_gammas = seg_gammas
        return seg_gammas
    
    def get_s_activity_segments(self):
        compseg = self.compseg
        seg_gamma_pr = np.ones((compseg, 2))
        seg_gamma_old_pr = np.ones((compseg, 2))
        counter = 0
        for l in np.arange(self.c.shape[1]):
            converge = np.ones((compseg, 1))
            AA = np.exp(-self.exchange_energy /self.rgas/self.T)*self.c[:,l]/self.acosmo[l]
            while np.max(converge) > self.thresh:
                seg_gamma_old_pr[:, l] = seg_gamma_pr[:,l]
                counter += 1
                seg_gamma_pr[:,l] = np.exp(-np.log(np.sum(AA*seg_gamma_old_pr[:,l], axis=1)))
                seg_gamma_pr[:,l] = (seg_gamma_pr[:,l] + seg_gamma_old_pr[:,l])/2
                converge = np.abs((seg_gamma_pr-seg_gamma_old_pr)/seg_gamma_old_pr)
                if counter > 1e3 or np.max(converge) < self.thresh:
                    break
        self.seg_gamma_pr = seg_gamma_pr
        return seg_gamma_pr
    
    def get_lngamma_SG(self):
        x1 = self.x1
        x2 = (1-self.x1)
        self.RNORM = np.array([0.0, 0.0])
        self.QNORM = np.array([0.0, 0.0])
        gammas1 = np.zeros(len(x1))
        gammas2 = np.zeros(len(x2))
        ln_gamma_sg1s = []
        ln_gamma_sg2s = []
        for i in range(self.c.shape[1]):
            self.RNORM[i] = self.vcosmo[i]/self.vnorm
            self.QNORM[i] = self.acosmo[i]/self.anorm

        for x in range(len(x1)):
            sum_gamma1 = 0
            sum_gamma2 = 0
            self.L1 = (self.coord/2)*(self.RNORM[0] - self.QNORM[0])-(self.RNORM[0] - 1)
            self.L2 = (self.coord/2)*(self.RNORM[1] - self.QNORM[1])-(self.RNORM[1] - 1)
            self.BOT_THETA = x1[x]*self.QNORM[0] + (1-x1[x])*self.QNORM[1]
            self.BOT_PHI = x1[x]*self.RNORM[0] + (1-x1[x])*self.RNORM[1]
            self.THETA1 = x1[x]*self.QNORM[0]/self.BOT_THETA
            self.THETA2 = (1-x1[x])*self.QNORM[1]/self.BOT_THETA
            self.PHI1 = x1[x]*self.RNORM[0]/self.BOT_PHI
            self.PHI2 = (1-x1[x])*self.RNORM[1]/self.BOT_PHI

            ln_gamma_sg1 = np.log(self.PHI1/x1[x]) + (self.coord/2)*self.QNORM[0] * \
                       np.log(self.THETA1/self.PHI1) + self.L1 - (self.PHI1/x1[x]) * (x1[x]*self.L1 + (1-x1[x])*self.L2)
            ln_gamma_sg2 = np.log(self.PHI2/(1-x1[x])) + (self.coord/2) * self.QNORM[1] * \
                       np.log(self.THETA2/self.PHI2) + self.L2 - (self.PHI2/(1-x1[x])) * (x1[x]*self.L1 + (1-x1[x])*self.L2)
            ln_gamma_sg1s.append(ln_gamma_sg1)
            ln_gamma_sg2s.append(ln_gamma_sg2)
        self.ln_gamma_sg1 = ln_gamma_sg1s
        self.ln_gamma_sg2 = ln_gamma_sg2s
        return ln_gamma_sg1s, ln_gamma_sg2s
    
    def get_input_array(self):
        row = np.hstack([np.concatenate([self.c1.reshape(-1,1), 
                              self.vcosmo[0].reshape(-1,1), 
                              self.acosmo[0].reshape(-1,1),
                              self.c2.reshape(-1,1),
                              self.acosmo[1].reshape(-1,1),
                              self.vcosmo[1].reshape(-1,1)]) for i in range(len(self.x1))])
        row2 = np.hstack([np.concatenate([self.c2.reshape(-1,1), 
                              self.acosmo[1].reshape(-1,1),
                              self.vcosmo[1].reshape(-1,1), 
                              self.c1.reshape(-1,1), 
                              self.acosmo[0].reshape(-1,1),
                              self.vcosmo[0].reshape(-1,1)]) for i in range(len(self.x1))])
        mol_frac = np.concatenate([self.x1, 1-self.x1])
        X = np.hstack((row, row2))
        X = np.vstack((X, mol_frac)).T
        return X
    
    def get_output_array(self):
        row = np.vstack([np.vstack([self.ln_gamma1, self.ln_gamma2])])
        row2 = np.vstack([np.vstack([self.ln_gamma2, self.ln_gamma1])])
        Y = np.concatenate((row, row2)).reshape(len(self.x1)*2, 2)
        return Y
    
    def get_input_df(self):
        X = self.get_input_array()
        sig_col1 = ['sigma_'+str(i)+'_1' for i in range(len(self.c1[0]))]
        sig_col2 = ['sigma_'+str(i)+'_2' for i in range(len(self.c2[0]))]
        vol_cols = ['Volume_1','Volume_2']
        cols = sig_col1
        cols.append(vol_cols[0])
        cols.extend(sig_col2)
        cols.append(vol_cols[1])
        df = pd.DataFrame(X)
        df = df.rename(columns=dict(zip(df.columns, cols)))
        return df
        
def sigma_mod(sigma, scale = 1, shift = 0, flip=False):
    # order 
    # 1. flip
    # 2. shift
    # 3. scale
    sigma = np.array(sigma)
    if flip==True:
        sigma = sigma[::-1]
    new_sigma = np.zeros(len(sigma))
    for idx in range(len(sigma)):
        new_idx = idx + shift
        if new_idx >= len(sigma):
            new_idx = new_idx - len(sigma)
        new_sigma[new_idx] = sigma[idx]
    new_sigma = new_sigma*scale
    return new_sigma
