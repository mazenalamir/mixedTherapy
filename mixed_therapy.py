'''
This module has been used to generate the results contained in the paper
Learning-based sensitivity analysis and feedback design for drug delivery
of mixed therapy of cancer in the presence of high model uncertainty
by Mazen Alamir

Author Mazen Alamir
CNRS, University of Grenoble Alpes
Date May 2022
'''

import numpy as np
import pandas as pd
import time, random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

#----- Global constants
dt = 1.0 / 24
Nsub_steps = 5

class Result:
    pass

class Context:

    def __init__(self, X0, df_p_model, sol):
        self.X0 = X0
        self.df_p_model = df_p_model
        self.sol = sol

def generate_x0(n_samples):
    """Random generation of n_samples of
    initial state for the cancer model
    generate_x0(n_samples)
    output : a listt of initial states.
    """
    Cmin = ModelParam().C_min
    # the power are given for a log10 scale.
    power_x_min = np.array([5, -3, -3, np.log10(1.05 * Cmin), -3, -3])
    power_x_max = np.array([9, 3, 8, 11.1, -3, -3])

    X0 = []
    for i in range(n_samples):
        dpower = power_x_max - power_x_min
        power_x = power_x_min + np.multiply(np.random.rand(6), dpower)
        x0 = np.array([10 ** power_x[i] for i in range(6)])
        X0.append(x0)
    return X0
#----
class ModelParam:

    def __init__(self):
        """The class that defines the nominal model parameters"""

        self.p_nom = {
            'a': 4.31e-1,
            'b': 1.02e-9,
            'c': 6.41e-11,
            'd': 2.34,
            'l': 2.08e0,
            's': 8.39e-2,
            'e': 2.08e-7,
            'f': 4.12e-2,
            'g': 1.25e-2,
            'h': 2.02e7,
            'p': 3.42e-6,
            'm': 2.04e-1,
            'j': 2.49e-2,
            'k': 3.66e7,
            'q': 1.42e-6,
            'r1': 1.1e-7,
            'r2': 6.5e-11,
            'u': 3.0e-10,
            'KT': 9.0e-1,
            'KN': 6.0e-1,
            'KL': 6.0e-1,
            'KC': 6.0e-1,
            'alpha': 7.5e8,
            'beta': 1.2e-2,
            'gamma': 9.0e-1,
            'pI': 1.25e-1,
            'gI': 2.0e7,
            'muI': 1e1
        }

        # Computation of the minimal value for the circulating lymphocytes

        self.C_min = self.p_nom['alpha'] / self.p_nom['beta'] / 2

    def generate_parameters(self,
                         mode='gaussian',
                         sigma=None,
                         alpha_minus=None,
                         alpha_plus=None,
                         n_samples=1):
        """Function that defines the disturbed parameters
        two modes are available: gaussian and uniform.
        generate_parameters(self,
                         mode='gaussian',
                         sigma=None,
                         alpha_minus=None,
                         alpha_plus=None,
                         n_samples=1)

        output : list of ModelParam objects with disturbed p_nom
        """
        list_of_p = []

        for _ in range(n_samples):

            pm = ModelParam()
            p = self.p_nom.copy()
            if mode.lower() == 'gaussian':
                for k in p:
                    p[k] = np.random.normal(p[k], abs(p[k]) * sigma)
            else:
                for k in p:
                    p_min = p[k] - alpha_minus * abs(p[k])
                    p_max = p[k] + alpha_plus * abs(p[k])
                    p[k] = p_min + np.random.rand() * (p_max - p_min)

            pm.p_nom = p.copy()
            list_of_p.append(pm)
        if n_samples == 1:
            pm.p_nom = p.copy()
            return pm
        else:
            return list_of_p
#----
class ControlParam:
    """The class that defines the parameters of the control strategy

    c_d : involved in the stopping condition for the immunotherapy delivery
        nominal value = 1, when higher, delivery might continue although useless
        when lower, imuno might be stopped while useful. Suggested interval (0.8, 1.5)

    T_max : denominateur of the expression T/T_max that multiply the drug injection.
        suggested interval (1e7,1e8)

    r : rate of decrease of the tumor involved in the condition dTdt < -r * T that fires the stop
        of the drug injection. suggested interval (0.1, 10)

    beta_C : security margin on the health constraints C >= beta_C * C_min It enables to meet the
        constraints by tightening this condition in spite of the parameters discrepancy. Suggested
        interval (1, 1.5)

    mu_C : gain of the constraints on the desired constraint violation dynamics
        dCdt >= mu_C*(beta_C * C_min -C) suggested interval (0.1, 1)

    T_stop : the threshold on the tumor size that fire the stopping of the drug delivery.
        suggested interval (1e3, 1e5)

    basic_period : the basic period including a treatment period followed by a rest period. suggested
        interval (0.5, 1, 2)

    treatment_period : The fraction of the basic period during which the feedback is active.
        suggested interval (0.25, 0.9) - should be in (0,1)
    """

    def __init__(self, d):

        self.c_d = d['c_d']
        self.T_max = d['T_max']
        self.r = d['r']
        self.beta_C = d['beta_C']
        self.mu_C = d['mu_C']
        self.T_stop = d['T_stop']
        self.treatment_duration = d['treatment_duration']
        self.basic_period = d['basic_period']
        #---------------------------
        self.vI_max = 1e4
        self.vL_max = 1e7
        self.vM_max = 1.0
        #---------------------------

    def generate_ctr_options(n_samples):
        """generate_ctr_options(n_samples) :
        generate a list of ControlParam"""

        generator = {
            'c_d': lambda n: 0.8 + np.random.rand(n) * (1.5 - 0.8),
            'T_max': lambda n: 10 ** (7 + np.random.rand(n) * (8 - 7)),
            'r': lambda n: 10 ** (-1 + np.random.rand(n) * (1 - (-1))),
            'beta_C': lambda n: 1 + np.random.rand(n) * (2 - 1),
            'mu_C': lambda n: 10 ** (-1 + np.random.rand(n) * (0 -(-1))),
            'T_stop': lambda n: 10 ** (1 + np.random.rand(n) * (3 - 1)),
            'basic_period': lambda n: [random.choice([0.5, 1, 2]) for _ in range(n)],
            'treatment_duration': lambda n: 0.2 + np.random.rand(n) * (0.9 - 0.2),
        }

        options = {}
        list_of_d = []
        for k in generator:
            options[k] = generator[k](n_samples)
        for i in range(n_samples):
            d = {k:options[k][i] for k in generator}
            list_of_d.append(d)

        list_of_p_ctr = [ControlParam(d) for d in list_of_d]

        return list_of_p_ctr
#----
def plot_traj(t, X, U, p_model):
    """ plot_traj(t, X, U, p_model)

    utility that plot the trajectories of
    states and control for a given simulated scenario"""

    fig, ax = plt.subplots(nrows=3, ncols=3,
                           figsize=(15, 11))

    color = 'k'

    titles = {
        (0, 0):'Tumor (T)',
        (0, 1): 'NL-cells (N)',
        (0, 2): 'CD8+T-cells (L)',
        (1, 0): 'Circulating Lymphocites (C)',
        (1, 1): 'Chemotherapy concentration (M)',
        (1, 2): '(IL-2) Immuno therapy drug concentration (I)',
        (2, 0): '(IL-2) Immunotherapy injection (vI)',
        (2, 1): '(TIL) Immunotherapy injection (vL)',
        (2, 2): 'Chemothrapy injection (vM)',
    }

    ax[0, 0].plot(t, X[:, 0], color)
    ax[0, 1].plot(t, X[:, 1], color)
    ax[0, 2].plot(t, X[:, 2], color)

    ax[1, 0].plot(t, X[:, 3], color)
    ax[1, 0].plot(t, np.ones(t.shape) * p_model.C_min, '-.')
    ax[1, 1].plot(t, X[:, 4], color)
    ax[1, 2].plot(t, X[:, 5], color)

    ax[2, 0].plot(t, U[:, 0], color)
    ax[2, 1].plot(t, U[:, 1], color)
    ax[2, 2].plot(t, U[:, 2], color)

    for i in range(3):
        for j in range(3):
            ax[i,j].set_title(titles[(i,j)])

    for i in range(3):
        for j in range(3):
            ax[i, j].grid(True)
            ax[i, j].set_xlim(t.min(), t.max())

    plt.show()
#----
def feedback(x, p_model, p_ctr):
    """feedback(x, p_model, p_ctr) where
    x  :the current state of the model
    p_model : the model parameters that are used by the controller.
    p_ctr : the parameters of the controller
    """
    T, N, L, C, M = x[0], x[1], x[2], x[3], x[4]
    LsTp = np.power(L/T, p_model.p_nom['l'])
    D = p_model.p_nom['d'] * LsTp/(p_model.p_nom['s'] + LsTp)
    dTdt = (p_model.p_nom['a']*(1-p_model.p_nom['b'] * T) -
            p_model.p_nom['c'] * N - D - p_model.p_nom['KT'] * (1-np.exp(-M))) * T
    if dTdt < - p_ctr.r * T:
        vI, vL, vM = 0.0, 0.0, 0.0
    else:
        D_max = p_model.p_nom['d'] * p_ctr.c_d
        factor = abs(D_max-D) / D_max * T / p_ctr.T_max
        vI = p_ctr.vI_max * min(1, factor)
        vL = p_ctr.vL_max * min(1, factor)
        #----
        M_bar = max(0, p_ctr.mu_C * (C-p_ctr.beta_C * p_model.C_min))
        vM = min(p_ctr.vM_max , p_model.p_nom['gamma'] * M_bar)
        if (T < p_ctr.T_stop) or (C < p_ctr.beta_C * p_model.C_min):
            vM = 0

    return vI, vL, vM
#----
def ode(x, u, p_model):
    """ode(x, u, p_model): the differential equations representing the model"""
    pm = p_model.p_nom
    T, N, L, C, M, I = x[0], x[1], x[2], x[3], x[4], x[5]
    if x.min() >= 0:
        vI, vL, vM = u[0], u[1], u[2]
        plT = np.power((L + 1e-8) / (T + 1e-8), pm['l'])
        D = pm['d'] * plT / (pm['s'] + plT)
        UnmeM = 1 - np.exp(-M)
        DT2 = D * D * T * T
        dTdt = pm['a'] * T * (1 - pm['b'] * T) - pm['c'] * N * T - D * T - pm['KT'] * UnmeM * T
        dNdt = pm['e'] * C - pm['f'] * N + pm['g'] * T * T / (pm['h'] + T * T) * N - pm[
            'p'] * N * T - pm['KN'] * UnmeM * N
        dLdt = -pm['m'] * L + pm['j'] * DT2 / (pm['k'] + DT2) * L - pm['q'] * L * T
        dLdt += (pm['r1'] * N + pm['r2'] * C) * T - pm['u'] * N * L * L - pm['KL'] * UnmeM * L
        dLdt += pm['pI'] * I / (pm['gI'] + I) * L + vL
        dCdt = pm['alpha'] - pm['beta'] * C - pm['KC'] * UnmeM * C
        dMdt = -pm['gamma'] * M + vM
        dIdt = -pm['muI'] * I + vI

        dxdt = np.array([dTdt, dNdt, dLdt, dCdt, dMdt, dIdt])
    else:
        dxdt = np.zeros(6)

    return dxdt
#----
def simulate_slot(x0, p_model, p_model_ctr, p_ctr, Tsim, mode):
    """simulate_slot(x0, p_model, p_model_ctr, p_ctr, Tsim, mode)
    simulate a period of time under control (mode="cl) or under no drug injection
    (mode="ol)
    x0 : the initial state of the model
    p_model : the true value of the model parameter object
    p_model_ctr : the presumed value (by the controller) of the model parameter
    p_ctr : the parameter of the controller
    Tsim : the time to be simulated (expressed in days)
    mode : define the closed-loop / open-loop mode for the controller
    output : tcl, Xcl, Ucl
    """
    Xcl = [x0]
    Ucl = []
    N_sim = int(Tsim / dt)
    tcl = np.array([i * dt / Nsub_steps for i in range(N_sim * Nsub_steps + 1)])
    for i in range(N_sim):
        if mode.lower() == 'cl':
            vI, vL, vM = feedback(Xcl[-1], p_model_ctr, p_ctr)
        else:
            vI, vL, vM = 0, 0, 0
        u = np.array([vI, vL, vM])
        for _ in range(Nsub_steps):
            xact = Xcl[-1]
            k1 = ode(xact, u, p_model)
            k2 = ode(xact + 0.5 * k1 * dt / Nsub_steps, u, p_model)
            k3 = ode(xact + 0.5 * k2 * dt / Nsub_steps, u, p_model)
            k4 = ode(xact + k3 * dt / Nsub_steps, u, p_model)
            Xcl += [xact + dt / Nsub_steps / 6 * (k1 + 2 * (k2 + k3) + k4)]
            Ucl.append(u)

    Xcl = np.array(Xcl)
    Ucl = np.array(Ucl)

    return tcl, Xcl, Ucl
#----
def simulate_basic_period(x0, p_model, p_model_ctr, p_ctr, plot):
    """simulate_basic_period(x0, p_model, p_model_ctr, p_ctr, plot)
    Simulate two slots (treatment period followed by a rest period)
    The duration of these slots are defined by the control parameters
    p_ctr.treatment_duration, p_ctr.basic_period. The rest period is
    the difference between the basic period and the treatment period
    """
    t, X, U = [], [], []
    feedback_period =  p_ctr.treatment_duration * p_ctr.basic_period
    rest_period = (1 - p_ctr.treatment_duration) * p_ctr.basic_period
    tcl, Xcl, Ucl = simulate_slot(x0, p_model, p_model_ctr, p_ctr, feedback_period, "cl")
    t.append(tcl[0:-1]); X.append(Xcl[0:-1,:]); U.append(Ucl)
    tcl, Xcl, Ucl = simulate_slot(Xcl[-1,:], p_model, p_model_ctr, p_ctr, rest_period, "ol")
    t.append(t[-1].max()+tcl[1:]); X.append(Xcl[0:-1,:]); U.append(Ucl)
    t = np.concatenate([tt for tt in t])
    X = np.concatenate([x for x in X])
    U = np.concatenate([u for u in U])

    if plot:
        plot_traj(t, X, U, p_model)

    return t, X, U
#----
def simulate(x0, p_model, p_model_ctr, p_ctr, n_periods, plot):
    """simulate(x0, p_model, p_model_ctr, p_ctr, n_periods, plot)
    simulate a number n_periods of basic periods, each of which contains a
    treatment period followed by a rest period.
    output : t, X, U, drug_usage
    """
    t, X, U = [], [], []
    tact = 0
    x_act = x0
    for  _ in range(n_periods):
        t1, X1, U1 = simulate_basic_period(x_act, p_model, p_model_ctr, p_ctr, False)
        t.append(tact + t1)
        X.append(X1)
        U.append(U1)
        tact = tact + t1.max()
        x_act = X1[-1, :]

    t = np.array(t).flatten()
    X = np.vstack(X)
    U = np.vstack(U)

    bounds = [p_ctr.vI_max, p_ctr.vL_max, p_ctr.vM_max]
    quantity = np.array([U[:, i].sum(axis=0) * dt / bounds[i] for i in range(3)])

    drug_usage = {
        'vI': quantity[0],
        'vL': quantity[1],
        'vM': quantity[2],
    }
    if plot:
        plot_traj(t, X, U, p_model)

    return t, X, U, drug_usage
#----
def generate_learning_data(file_name, Tsim, n_samples, mode, alpha_minus, alpha_plus, sigma):
    """generate_learning_data(file_name, Tsim, n_samples,
                                mode, alpha_minus, alpha_plus, sigma)

    file_name : the result is saved is stored in the file "file_name.pkl"
    Tsim : the simulated time in days
    n_samples : the number of samples in the learning data
    mode : uniform or gaussian regarding the model's parameters dispersion
    alpha_minus : the lower bound in percentage of the nominal value (applies only in uniform mode)
    alpha_plus : the upper bound in percentage of the nominal value (applies only in uniform mode)
    sigma : the standard deviatin expressed in fraction of the nominal values.
    output : a Result object is stored in the file_name.pkl with the following fields
    t, X, U, p_model, p_ctr, drug_usage.
    """
    p_model_ctr = ModelParam()
    create_learning_data = True

    if create_learning_data:

        list_x0  = generate_x0(n_samples)
        list_p_model = ModelParam().generate_parameters(mode, sigma, alpha_minus, alpha_plus, n_samples)
        list_p_ctr = ControlParam.generate_ctr_options(n_samples)
        R = []
        for i in tqdm(range(n_samples)):
            x0, p_model, p_ctr = list_x0[i], list_p_model[i], list_p_ctr[i]
            n_periods = int(Tsim / p_ctr.basic_period)
            t, X, U, drug_usage = simulate(x0, p_model, p_model_ctr, p_ctr, n_periods, False)
            Rsim = Result()
            Rsim.t = t
            Rsim.X = X
            Rsim.U = U
            Rsim.p_model = p_model
            Rsim.p_model_ctr = p_model_ctr
            Rsim.p_ctr = p_ctr
            Rsim.drug_usage = drug_usage
            R.append(Rsim)

        pickle.dump(R, open(f'{file_name}.pkl', 'wb'))

        return R
#----
def create_dataframe_from_result(R):
    """
    create_dataframe_from_result(R)
    create a data frame from the output of generate_learning_data utility.

    the columns of the dataframe are:

    ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'a', 'b', 'c', 'd', 'l', 's', 'e',
       'f', 'g', 'h', 'p', 'm', 'j', 'k', 'q', 'r1', 'r2', 'u', 'KT', 'KN',
       'KL', 'KC', 'alpha', 'beta', 'gamma', 'pI', 'gI', 'muI', 'c_d', 'T_max',
       'r', 'beta_C', 'mu_C', 'T_stop', 'Treatment_duration', 'basic_period',
       'Tf', 'lower_C'

    output : the dataframe df
    """

    n_samples = len(R)
    df = []
    for i in range(n_samples):
        r = R[i]
        x0 = r.X[0, :]
        p_model = r.p_model
        p_ctr = r.p_ctr
        drug_usage = r.drug_usage
        final_tumor = r.X[-1, 0]
        lower_C = (r.X[:, 3].min() - p_model.C_min)/p_model.C_min
        dfi_x = pd.DataFrame(x0, index=[f'x{i + 1}' for i in range(len(x0))]).T
        p_model_values = []
        p_model_keys = []
        for k in p_model.p_nom:
            p_model_keys += [k]
            p_model_values += [p_model.p_nom[k]]

        p_ctr_index = ['c_d', 'T_max', 'r', 'beta_C', 'mu_C',
                       'T_stop', 'Treatment_duration', 'basic_period']

        p_ctr_values = [p_ctr.c_d, p_ctr.T_max, p_ctr.r,
                        p_ctr.beta_C, p_ctr.mu_C, p_ctr.T_stop,
                        p_ctr.treatment_duration, p_ctr.basic_period]

        dfi_p_model = pd.DataFrame(p_model_values, index=p_model_keys).T
        dfi_p_ctr = pd.DataFrame(p_ctr_values, index=p_ctr_index).T
        dfi_final_Tumor = pd.DataFrame([final_tumor], index=['Tf']).T
        dfi_lower_C = pd.DataFrame([lower_C], index=['lower_C']).T
        dfi_drug_usage_I = pd.DataFrame([drug_usage['vI']], index=['vI_usage']).T
        dfi_drug_usage_L = pd.DataFrame([drug_usage['vL']], index=['vL_usage']).T
        dfi_drug_usage_M = pd.DataFrame([drug_usage['vM']], index=['vM_usage']).T
        dfi = pd.concat([dfi_x, dfi_p_model, dfi_p_ctr,
                         dfi_final_Tumor, dfi_lower_C,
                         dfi_drug_usage_I, dfi_drug_usage_L, dfi_drug_usage_M], axis=1)

        df.append(dfi)

    df = pd.concat(df, axis=0).reset_index(drop=True)
    return df
#----
def create_df_p_model(n_samples, mode, sigma, alpha):
    """create_df_p_model(n_samples, mode, sigma, alpha)
    Create a dataframe that contains the randomly sampled parameter
    to be used in the definition of the cost function. mode define the uniform vs guassian
    randomness, sigma for the gaussian, alpha for the uniform (only uniform is used in the paper)"""
    mp = ModelParam()
    list_p_model = mp.generate_parameters(mode=mode,
                                          sigma=sigma,
                                          alpha_minus=alpha,
                                          alpha_plus=alpha,
                                          n_samples=n_samples)
    df = []
    for i in range(len(list_p_model)):
        # model's parameter
        p_model = list_p_model[i]
        p_model_values = []
        p_model_keys = []
        for k in p_model.p_nom:
            p_model_keys += [k]
            p_model_values += [p_model.p_nom[k]]
        dfi = pd.DataFrame(p_model_values, index=p_model_keys).T

        df.append(dfi)

    df = pd.concat(df, axis=0).reset_index(drop=True)

    return df
#----
def predict_over_cloud(xp, c):
    """cost(xp, c)
    the creation is based on the following convention for xp
    which is unknown control parameter:
    r, beta_C, Treatment_duration, T_stop = *xp
    c.x0 is the initial stat while c.df_p_model is the data frame of
    the model parameters randomly generated using the function
    create_df_p_model described above.
    THIS FUNCTION IS NOTE USED IN THE PAPER RESULTS
    """
    df_p_model = c.df_p_model
    n_samples = len(df_p_model)

    df_x = pd.DataFrame(c.X0, columns=[f'x{i + 1}' for i in range(len(c.X0[0]))])

    df_ctr = pd.DataFrame(np.array(list(xp) * n_samples).reshape(-1, len(xp)),
                          columns=['r', 'beta_C', 'Treatment_duration', 'T_stop'])

    df = pd.concat([df_x, df_p_model, df_ctr], axis=1)
    health_condition = c.sol.cl_health_selected.predict(df[c.sol.colX_health_important].values)
    contraction_condition = c.sol.cl_contraction_selected.predict(df[c.sol.colX_contraction_important].values)
    Quantity_M = c.sol.reg_M.predict(df[c.sol.colX_both].values)
    Quantity_I = c.sol.reg_I.predict(df[c.sol.colX_both].values)
    Quantity_L = c.sol.reg_L.predict(df[c.sol.colX_both].values)

    R = Result()
    R.health_condition = health_condition
    R.contraction_condition = contraction_condition
    R.QM = Quantity_M
    R.QI = Quantity_I
    R.QL = Quantity_L
    return  R