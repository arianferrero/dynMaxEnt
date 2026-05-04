"""
Boltzmann-biased ACHR sampler for COBRApy 
==========================================

This module implements a Metropolis–Hastings ACHR sampler with a
Boltzmann-style bias toward selected reaction fluxes.

Main features:
- Per-reaction beta (β) weighting
- Optional automatic beta scaling via pilot sampling
- Optional annealing schedule (gradual activation of bias)
- Fully compatible with COBRApy ACHR framework
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde
import math
import random as rd
import cobra
from cobra.sampling.achr import ACHRSampler
from cobra.sampling.core import step


class BiasedSampler(ACHRSampler):
    """Boltzmann-biased ACHR sampler with multiple β, auto-scaling and annealing."""

    def __init__(
        self,
        model,
        reactions,
        betas=1.0,
        thinning=100,
        nproj=None,
        seed=None,
        auto_scale=True,
        pilot_samples=300,
        anneal_steps=0,
        verbose=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : cobra.Model
            COBRA model to sample.
        reactions : list of str
            Target reaction IDs.
        betas : float or list of float
            Bias strengths (one per reaction or scalar).
        auto_scale : bool, optional
            If True, scale betas by 1/std(target flux) estimated from a short
            unbiased pilot run (default True).
        pilot_samples : int, optional
            Number of pilot samples for scaling (default 300).
        anneal_steps : int, optional
            Number of sampling iterations over which to ramp β from 0 to full
            value (default 0 means no annealing).
        verbose : bool, optional
            Print diagnostic info (default True).
        """
        super().__init__(model, thinning=thinning, nproj=nproj, seed=seed, **kwargs)
        self.target_rxns = [model.reactions.get_by_id(r) for r in reactions]

        if np.isscalar(betas):
            betas = [float(betas)] * len(reactions)
        elif len(betas) != len(reactions):
            raise ValueError("`betas` must be scalar or same length as `reactions`")

        self.base_betas = np.array(betas, dtype=float)
        self.betas = self.base_betas.copy()
        self.anneal_steps = int(anneal_steps)
        self.verbose = verbose

        # Map reactions to indices
        rxn_ids = [r.id for r in model.reactions]
        self.target_idx = np.array([rxn_ids.index(r.id) for r in self.target_rxns])

        # Optionally scale betas by inverse std from pilot run
        if auto_scale:
            if verbose:
                print(f"[BiasedSampler] Running pilot sampling ({pilot_samples} samples)...")
            pilot = self._pilot_sample(pilot_samples)
            stds = pilot.std(axis=0)
            stds[stds == 0] = 1.0
            self.betas = self.base_betas / stds
            if verbose:
                print("[BiasedSampler] Pilot stds:",
                      {r.id: round(s, 3) for r, s in zip(self.target_rxns, stds)})
                print("[BiasedSampler] Scaled betas:",
                      {r.id: round(b, 3) for r, b in zip(self.target_rxns, self.betas)})

        self.accepted = 0
        self.trials = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pilot_sample(self, n=200):
        """Run a short unbiased ACHR sampling to estimate stds."""
        from cobra.sampling.achr import ACHRSampler
        tmp = ACHRSampler(self.model, thinning=self.thinning, seed=self._seed + 1)
        pilot_df = tmp.sample(n)
        return pilot_df[[r.id for r in self.target_rxns]]

    def _log_weight(self, fluxes):
        """Compute log weight (sum beta_i * v_i) for numerical stability."""
        return float(np.dot(self.betas, fluxes[self.target_idx]))

    def _current_beta_scale(self):
        """If annealing, compute current scaling factor in [0,1]."""
        if self.anneal_steps <= 0 or self.n_samples >= self.anneal_steps:
            return 1.0
        return self.n_samples / max(1, self.anneal_steps)

    # ------------------------------------------------------------------
    # Single MH iteration
    # ------------------------------------------------------------------

    def __single_iteration(self):
        pi = np.random.randint(self.n_warmup)
        delta = self.warmup[pi, :] - self.center
        candidate = step(self, self.prev, delta)

        flux_prev = self.prev[self.fwd_idx] - self.prev[self.rev_idx]
        flux_cand = candidate[self.fwd_idx] - candidate[self.rev_idx]

        scale = self._current_beta_scale()
        log_old = scale * self._log_weight(flux_prev)
        log_new = scale * self._log_weight(flux_cand)
        log_acc = log_new - log_old

        self.trials += 1
        if log_acc >= 0 or np.log(np.random.rand()) < log_acc:
            self.prev = candidate
            self.accepted += 1

        self.center = (self.n_samples * self.center + self.prev) / (self.n_samples + 1)
        self.n_samples += 1

    # ------------------------------------------------------------------
    # Public sampling interface
    # ------------------------------------------------------------------

    def sample(self, n: int, fluxes: bool = True) -> pd.DataFrame:
        """Generate biased flux samples with optional annealing."""
        samples = np.zeros((n, self.warmup.shape[1]))

        for i in range(1, self.thinning * n + 1):
            self.__single_iteration()
            if i % self.thinning == 0:
                samples[i // self.thinning - 1, :] = self.prev

        acc_rate = self.accepted / max(1, self.trials)
        if self.verbose:
            print(f"[BiasedSampler] Acceptance rate: {acc_rate:.3f}")
            print(f"[BiasedSampler] Final β effective scale: {self._current_beta_scale():.2f}")

        if fluxes:
            names = [r.id for r in self.model.reactions]
            return pd.DataFrame(
                samples[:, self.fwd_idx] - samples[:, self.rev_idx],
                columns=names,
            ), self.betas
        else:
            names = [v.name for v in self.model.variables]
            return pd.DataFrame(samples, columns=names) , self._current_beta_scale()
        

        
            
def calculate_avg_curve(data):
    stats = {'t':list(data.t), 'avg':[], 'sd':[]}
    
    for t in data.iterrows():
        stats['avg'].append(np.mean(t[1][1:]))
        stats['sd'].append(np.std(t[1][1:]))
    
    avg_df = pd.DataFrame(stats)
    
    return avg_df


def get_growth_from_OD(time, od, a=100):
    """
    Compute growth rate from OD measurements using moving average smoothing.
    
    Parameters
    ----------
    time : array-like
        Time points
    od : array-like
        Optical density values
    a : int
        Window size for moving average (default = 100)
    
    Returns
    -------
    time : ndarray
        Time points (same length as input)
    GR1 : ndarray
        Smoothed growth rate
    """

    time = np.array(time)
    od = np.array(od)

    # Time step
    delta_t = np.mean(np.diff(time))

    # Log-transformed signal
    S = np.log(od * 0.35)

    # Moving average (same length as input)
    kernel = np.ones(a) / a
    S1 = np.convolve(S, kernel, mode="same")

    # Growth rate (derivative)
    GR = np.gradient(S1, delta_t)

    # Smooth growth rate
    GR1 = np.convolve(GR, kernel, mode="same")

    return time, GR1



def get_uptake_from_conc(time, od, conc, a = 100):
    """
    Compute growth rate from OD measurements using moving average smoothing.
    
    Parameters
    ----------
    time : array-like
        Time points
    od : array-like
        Optical density values
    a : int
        Window size for moving average (default = 100)
    
    Returns
    -------
    time : ndarray
        Time points (same length as input)
    GR1 : ndarray
        Smoothed growth rate
    """

    time = np.array(time)
    od = np.array(od)
    conc = np.array(conc) / 59 * 1000

    # Time step
    delta_t = np.mean(np.diff(time))

    # Log-transformed signal
    S = np.log(conc)

    # Moving average (same length as input)
    kernel = np.ones(a) / a
    S1 = np.convolve(S, kernel, mode="same")

    # Growth rate (derivative)
    uptake = np.gradient(S1, delta_t) / (od*0.35)

    # Smooth growth rate
    uptake1 = np.convolve(uptake, kernel, mode="same")

    return time, uptake1
    



def michaelis_menten(K, V, C):
    """
    Compute flux using Michaelis-Menten kinetics.

    Parameters
    ----------
    K : float
        Michaelis constant (substrate affinity).
    V : float
        Maximum uptake rate (Vmax).
    C : float
        Substrate concentration.

    Returns
    -------
    flux : float
        Uptake flux at concentration C.
    """
    return V * C / (K + C)



def find_betas_warmup(avg_g, first=True, beta_lambda=None):

    if first: 
        reference_file = 'results_cobra_sweep_betas_experiment_conditions_glc_ac.dat'
        reference_betas = pd.read_csv(reference_file, sep='\t')
    
        diff = math.inf
        for params in reference_betas.iterrows():
            diff_i = ((params[1]['mean_lambda']-avg_g)/params[1]['sd_lambda'])**2 
                    
            if diff_i < diff and params[1]['beta_lambda']!=0: 
                diff = diff_i
                beta_lambda = params[1]['beta_lambda'] 
    else: 
        beta_lambda = beta_lambda + beta_lambda*np.random.uniform(-0.5,0.5)
       
    return beta_lambda
     
    

def biased_sampling(model, reactions_betas, n_samples = 2000, anneal_steps = 2000, pilot_samples = 1000, 
                    verbose = False):
    # perform biased sampling
    reactions = list(reactions_betas.keys())
    betas = list(reactions_betas.values())
    sampler = BiasedSampler(model, reactions=reactions, betas=betas,
                            anneal_steps=anneal_steps, pilot_samples = pilot_samples, verbose=verbose)
    samples, scaled_betas = sampler.sample(n_samples)
    
    return samples, scaled_betas


def find_betas_and_sample(avg_g,model, n_samples = 2000, anneal_steps = 2000, pilot_samples = 1000, 
                verbose = False, first=False, prev_beta_lambda=None):
    beta_lambda  = find_betas_warmup(avg_g, first=first, beta_lambda=prev_beta_lambda)
    reactions_betas = {'Biomass_Ecoli_core':beta_lambda}
    sample, scaled_betas = biased_sampling(model, reactions_betas, n_samples = 2000, 
                                               anneal_steps = 2000, pilot_samples = 1000, 
                                               verbose = False)
    
    rel_err_lambda = (avg_g-np.mean(sample['Biomass_Ecoli_core'])) / avg_g
    
    iterations_lambda = 0
    while np.abs(rel_err_lambda) > 0.1 :
    
        if iterations_lambda >=10:
            if rel_err_lambda/new_rel_err_lambda < 0: 
                new_beta_lambda = beta_lambda - np.abs(beta_lambda)*rel_err_lambda*np.random.random()
            else: 
                new_beta_lambda = beta_lambda + np.abs(beta_lambda)*rel_err_lambda*np.random.uniform(1,3)

            if iterations_lambda > 12: 
                iterations_lambda = 0
        else:
            new_beta_lambda = beta_lambda + np.abs(beta_lambda)*rel_err_lambda*np.random.random()

        print(new_beta_lambda, np.mean(sample['Biomass_Ecoli_core']), np.mean(sample['EX_ac_e']), avg_g)


        reactions_betas = {'Biomass_Ecoli_core':new_beta_lambda}
        sample, scaled_betas = biased_sampling(model, reactions_betas, n_samples = 2000, 
                                               anneal_steps = 2000, pilot_samples = 1000, 
                                               verbose = False)

        new_rel_err_lambda = (avg_g - np.mean(sample['Biomass_Ecoli_core'])) / avg_g
        if np.abs(new_rel_err_lambda) < np.abs(rel_err_lambda):
            beta_lambda = new_beta_lambda
            rel_err_lambda = new_rel_err_lambda
        else: 
            iterations_lambda += 1
           
                       
    return beta_lambda, sample



    
  
               
data = pd.read_csv('Ecoli_GlcAct_OD_RawData_012925/01292025.dat', sep='\t')
avg_df = calculate_avg_curve(data[['t', 'OD5', 'OD8']])

ini = 0
fin = -1

growth_df = {}
time_corrected, GR = get_growth_from_OD(np.array(avg_df.t[ini:fin]), np.array(avg_df['avg'][ini:fin]))
if 't' not in growth_df.keys():
    growth_df['t'] = time_corrected
growth_df['avg'] = GR
growth_df = pd.DataFrame(growth_df)

T = list(growth_df.t[100:])
growth_rates = list(growth_df.avg[100:])         


reference_betas = pd.read_csv('results_cobra_sweep_betas_experiment_conditions_glc_ac.dat', sep='\t')


# =====================================================================
# Simulation setup
# =====================================================================

# Experimental growth curve (must be defined beforehand)
exp_growths = growth_rates
time_points = np.linspace(0, len(exp_growths), 120).astype(int)

# Michaelis-Menten uptake parameters (glucose, acetate)
Kg, Vg = 0.001, 12
Ka, Va = 0.07, 8
oxygen = -17

# External metabolites tracked
acids = ['EX_ac_e', 'EX_acald_e', 'EX_akg_e', 'EX_etoh_e', 'EX_fum_e',
         'EX_lac__D_e', 'EX_glc__D_e', 'EX_pyr_e', 'EX_succ_e']

# Initial conditions
Cg_ini = 1      # initial glucose concentration (g/L)
Ca_ini = 0.2    # initial acetate concentration (g/L)
N_ini = avg_df.avg[100] * 0.35  # initial biomass density

# Flag: start new simulation or continue from saved results
folder = 'results/maxent_COBRA_g_fit/'


results_track = {'time': [],
                 'max_growth':[], 
                 'beta_lambda':[], 
                 'exp_growth':[], 
                 'cell_density':[], 
                 'avg_g':[]}

# =====================================================================
# Initialization (new or continuing run)
# =====================================================================
Cg = Cg_ini / 180 * 1000
Ca = Ca_ini / 59 * 1000
N = N_ini

ini = 0
# Initialize metabolite concentrations
concentrations = {a: [Ca if a == 'EX_ac_e' else (Cg if a == 'EX_glc__D_e' else 0)]
                  for a in acids}


model = cobra.io.load_json_model('ecoli_ACS.json')
reactions = [r.id for r in model.reactions]
# Apply initial constraints
if Cg < 0.001:
    model.reactions.ACS.upper_bound = 1000
else:
    model.reactions.ACS.upper_bound = 0
model.reactions.EX_glc__D_e.lower_bound = -michaelis_menten(Kg, Vg, Cg)
model.reactions.EX_ac_e.lower_bound = -michaelis_menten(Ka, Va, Ca)
model.reactions.EX_o2_e.lower_bound = oxygen
#model.reactions.ATPM.bounds = (4, 4)
model.reactions.EX_co2_e.lower_bound = 0
model.solver = 'glpk'
fba = model.optimize().objective_value

# =====================================================================
# Dynamic simulation loop
# =====================================================================

for iteration, i in enumerate(time_points[ini:-1]):
    first=False
    if iteration == 0: 
        first=True
        prev_beta_lambda=None
    else: 
        prev_beta_lambda = beta_lambda

    results_track['cell_density'].append(N)  

    avg_g = exp_growths[i]   # experimental growth rate

    lb = model.reactions.EX_glc__D_e.lower_bound
    while avg_g > fba: 
        lb -= 0.1
        model.reactions.EX_glc__D_e.lower_bound = lb 
        fba = model.optimize().objective_value


    results_track['max_growth'].append(fba)

    print(f"Experimental growth = {avg_g}")

    t = T[i]
    d_t = T[time_points[ini + iteration + 1]] - t

    results_track['time'].append(t)
    results_track['exp_growth'].append(avg_g)
    
   
    beta_lambda,  s = find_betas_and_sample(avg_g, model, 
                                                          n_samples = 2000, 
                                                          anneal_steps = 2000, 
                                                          pilot_samples = 1000, 
                                                          verbose = False, first=first,
                                                          prev_beta_lambda=prev_beta_lambda)

    # Save flux samples
    file = folder + f't{ini + iteration}_200samp.dat'
    print(f"Saving flux samples -> {file}")
    s.to_csv(file, sep='\t', index=False)
    results_track['beta_lambda'].append(beta_lambda)

    avg_g_model = np.mean(s['Biomass_Ecoli_core'])
    results_track['avg_g'].append(avg_g_model)


    # Update extracellular concentrations
    Ca += np.mean(s['EX_ac_e']) * N * d_t
    Cg += np.mean(s['EX_glc__D_e']) * N * d_t
    Cg, Ca = max(0, Cg), max(0, Ca)

    for a in acids:
        if a == 'EX_ac_e':
            concentrations[a].append(Ca)
        elif a == 'EX_glc__D_e':
            concentrations[a].append(Cg)
        else:
            concentrations[a].append(max(0, concentrations[a][ini + iteration] + np.mean(s[a]) * N * d_t))

    # Update biomass
    N = N * math.exp(np.mean(s['Biomass_Ecoli_core']) * d_t)

    # Reload model and reapply constraints
    model = cobra.io.load_json_model('ecoli_ACS.json')
    if Cg < 0.001:
        model.reactions.ACS.upper_bound = 1000
    else:
        model.reactions.ACS.upper_bound = 0
    model.reactions.EX_o2_e.lower_bound = oxygen
    model.reactions.EX_co2_e.lower_bound = 0
    model.solver = 'glpk'
    for a in acids:
        if a == 'EX_glc__D_e':
            model.reactions.get_by_id(a).lower_bound = -michaelis_menten(Kg, Vg, concentrations[a][ini + iteration + 1])
        else:
            model.reactions.get_by_id(a).lower_bound = -michaelis_menten(Ka, Va, concentrations[a][ini +iteration + 1])

    fba = model.optimize().objective_value

    # Save results
    pd.DataFrame(results_track).to_csv(folder+'cell_density.dat', sep='\t', index=False)
    pd.DataFrame(concentrations).to_csv(folder+'concentrations.dat', sep='\t', index=False)

    



