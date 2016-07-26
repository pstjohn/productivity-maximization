import scoop
from scoop import futures

import numpy as np
import pandas as pd
from Models.ecoli import ecoli_core_biomass

model = ecoli_core_biomass()
model.optimize(minimize_absolute_flux=1.0)

try:
    efms = pd.read_pickle('efm_hull_reduced.p')
except IOError:
    import os
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    efms = pd.read_pickle(curr_dir + '/efm_hull_reduced.p')

efms = efms.divide(-efms.EX_glc_e, 0)
# efms.drop(['EX_pyr_e'], 1, inplace=True)
# efms.drop(['EX_for_e'], 1, inplace=True)
# efms.drop(['EX_ac_e'], 1, inplace=True)

boundary_species = ['biomass', 'atpm', 'o2_e', 'glc__D_e', 'succ_e', 'ac_e', 'for_e']


# Parameters from [1]	T. J. Hanly and M. A. Henson, 2010
# v_max = -10.5
# Kg = (0.0027 / model.metabolites.glc__D_e.formula_weight) * 1000.

# Lin 2001
# yield_coeff = model.optimize().f / model.reactions.EX_glc_e.x
v_max = .5 / -0.043602107916
c_s_star = 155. / model.metabolites.glc__D_e.formula_weight * 1000.
k_s = 2.03 / model.metabolites.glc__D_e.formula_weight * 1000.
n = 0.603

c_p_star = [104.2 / 270.143 * 1000., # succ
            16.0  / 68.007 * 1000.,  # for
            44.2 / 82.034 * 1000.,   # ac
            # 74.1 / 110.044 * 1000.,  # pyr
            ]

def lin2008_uptake(x, v_max=v_max, c_s_star=c_s_star, k_s=k_s, n=n,
                   c_p_star=c_p_star):
    """Function which defines the maximum glucose uptake rate as a function of
    substrate and product concentrations"""

    c_s = x['glc__D_e']
    c_p = [x['succ_e'], x['for_e'], x['ac_e']]

    substrate_inhib =  v_max * ((1 - c_s/c_s_star)**n) * (c_s / (c_s + k_s))
    product_inhib = 1.
    for cp, cpi in zip(c_p, c_p_star):
        product_inhib *= (1 - cp/cpi)

    # The lower bound is negative since the yield_coeff is negative
    return (substrate_inhib * product_inhib, None)


atpm_req = model.reactions.ATPM.lower_bound
# atpm_req = 4.701

def atp_maintenance(x, atpm_req=atpm_req):
    """Function to enforce the output of ATP"""
    return (atpm_req, None)

o2_max = model.reactions.EX_o2_e.lower_bound
def o2uptake(x, atpm_req=atpm_req):
    """Function to limit the oxygen consumption """
    return (o2_max, None)

uptake_kinetics = {
    'glc__D_e' : lin2008_uptake,
    'atpm' : atp_maintenance,
    'o2_e' : o2uptake,
}

from Collocation.EFMcollocation import EFMcollocation

def setup_collocation(stage_breakdown):
    
    scoop.logger.debug('Collocation: started: {}'.format(stage_breakdown))
    coll = EFMcollocation(model, boundary_species, efms, stage_breakdown)
    scoop.logger.debug('Collocation: __init__ complete')
    coll.d = 3
    coll.setup(pvars={'alpha' : (1),})
    scoop.logger.debug('Collocation: setup complete')

    # Set up bounds
    coll.var.x_ub[:] = 1000.
    coll.var.x_in[:] = 0.1

    coll.var.x_in[0,0,:] = 1.

    # Biomass initial condition
    coll.var.x_lb[0,0,0] = 0.
    coll.var.x_ub[0,0,0] = 0.1
    coll.var.x_ub[:,:,0] = 15.
    coll.var.x_in[0,0,0] = 0.1

    # ATP maintenance initial condition
    coll.var.x_lb[0,0,1] = 0.
    coll.var.x_ub[0,0,1] = 0.
    coll.var.x_in[0,0,1] = 0.

    coll.var.h_lb[:] = 1.
    coll.var.h_ub[:] = 30.

    # Allow O2 levels to start at 0 and be negative
    coll.var.x_lb[:,:,2] = -1E6

    coll.var.x_lb[0,0,2] = 0.
    coll.var.x_ub[0,0,2] = 0.
    coll.var.x_in[0,0,2] = 0.

    coll.add_constraint(coll.xf['glc__D_e'] / coll.x0['glc__D_e'], lb=0., ub=.2,
                        msg='Minimum usage requirement')

    coll.add_boundary_constraints(uptake_kinetics)
    scoop.logger.debug('Collocation: kinetics complete')

    return coll

def calc_max_productivity(coll):

    coll.objective_sx = -((coll.xf['succ_e'] - coll.x0['succ_e']) /
                          coll.var.tf_sx)
    coll.initialize(max_iter=5000, print_level=0, print_time=False,
                    max_cpu_time=3600)
    coll.solve()
    return reduce_fn(coll)

def calc_max_yield(coll):

    coll.objective_sx = -((coll.xf['succ_e'] - coll.x0['succ_e']) /
                          (coll.x0['glc__D_e'] - coll.xf['glc__D_e']))
    coll.initialize(max_iter=5000, print_level=0, print_time=False,
                    max_cpu_time=3600)
    coll.solve()
    return reduce_fn(coll)

def initialize_collocation(coll):
    
    # Yield objective (alpha = 0)
    coll.objective_sx = -((coll.xf['succ_e'] - coll.x0['succ_e']) /
                          (coll.x0['glc__D_e'] - coll.xf['glc__D_e']))

    # Yield objective (alpha = 1)
    prod_sx = (((coll.xf['succ_e'] - coll.x0['succ_e']) / coll.var.tf_sx) -
               coll.pvar.alpha_sx[0])
    coll.add_constraint(prod_sx, lb=0., ub=0., msg='Prod constraint')
    coll.initialize(max_iter=5000, print_level=0, print_time=False,
                    max_cpu_time=3600)

    return coll

def reduce_fn(coll):
    out = {met + '_out' : x for met, x in 
           zip(coll.boundary_species, coll.var.x_op[-1, -1])}
    
    out.update({met + '_in' : x for met, x in 
           zip(coll.boundary_species, coll.var.x_op[0, 0])})

    out.update({'tf' : coll.var.tf_op})
    out.update({'h_{}'.format(i) : h for i,h in enumerate(coll.var.h_op)})

    # Calculate succinate yield and productivity
    out['yield_mol'] = (out['succ_e_out'] - out['succ_e_in']) / (
        out['glc__D_e_in'] - out['glc__D_e_out'])
    out['productivity_mol'] = (out['succ_e_out'] - out['succ_e_in']) / (out['tf'])

    # Add the maximum theoretical yield case
    # yields_mol.loc[1.] = max_yield
    # productivity_mol.loc[1.] = 0.

    out['yield_mass'] = (out['yield_mol'] *
                         model.metabolites.succ_e.formula_weight /
                         model.metabolites.glc__D_e.formula_weight)

    out['productivity_mass'] = (out['productivity_mol'] *
                                model.metabolites.succ_e.formula_weight / 
                                1000)

    return pd.Series(out)


# coll = setup_collocation([20])
# coll = setup_collocation([10, 10])
# coll = setup_collocation([7, 7, 7])
# coll = setup_collocation([4, 4, 4, 4, 4])
def run_alpha(alpha):
    try:
        coll.pvar.alpha_in[:] = alpha
        coll.solve()
        scoop.logger.info('Finished alpha={}'.format(alpha))
        return reduce_fn(coll)
    except RuntimeWarning:
        return pd.Series(np.nan)


if __name__ == "__main__":
    
    coll = setup_collocation([1]*20)

    max_prod = calc_max_productivity(coll)
    max_yield = calc_max_yield(coll)

    coll = initialize_collocation(coll)


