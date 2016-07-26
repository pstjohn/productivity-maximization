import bisect

import numpy as np
from scipy.interpolate import lagrange
import pandas as pd

import casadi as cs

from .VariableHandler import VariableHandler
from .BaseCollocation import BaseCollocation

class EFMcollocation(BaseCollocation):

    def __init__(self, model, boundary_species, efms, stage_breakdown):
        """ A class to handle the dynamic optimization based method of dynamic
        flux analysis from [1].

        model: a cobra.Model object.
            Contains the model to be optimized, reaction bounds and objective
            vectors will be preserved.

        boundary_species: a list
            Contains a list of strings specifying the boundary species which
            will be explicity tracked in the dynamic optimization. Biomass
            should be the first entry.

        efms: a pandas dataframe
            efms should be a dataframe containing compressed, boundary EFMs
            (with EFMs as columns) 

        stage_breakdown: list of ints
            Describes the number of finite elements to allocate to each of `n`
            fermentation divisions. The number of allowed fermentation
            divisions will be implicitly stated by the length of the passed
            list.

        """

        self.nx = len(boundary_species)
        self.nv = efms.shape[0]
        self.nf = len(stage_breakdown)

        # store stage breakdowns
        self.stage_breakdown = stage_breakdown

        # Handle boundary reactions
        self.boundary_species = boundary_species
        all_boundary_rxns = model.reactions.query('system_boundary', 'boundary')

        self.boundary_rxns = []
        for bs in boundary_species:
            rxns = all_boundary_rxns.query(lambda r: r.reactants[0].id == bs)

            assert len(rxns) == 1, (
                "Error finding boundary reactions for {}: ".format(bs) + 
                "{:d} reactions found".format(len(rxns)))

            self.boundary_rxns += [rxns[0].id]

        # Ensure all of the boundary species are accounted for
        present = pd.Series(self.boundary_rxns).isin(efms.columns)
        assert present.all(), ('EFMS not found for {}'.format(
            pd.Series(self.boundary_species)[~present].values))

        # Assure that efms are in the correct order
        self.efms_float = efms.loc[:, self.boundary_rxns]
        self.efms_object = pd.DataFrame(self.efms_float, dtype=object)

        super(EFMcollocation, self).__init__()

        # The finite element count must be set after initializing the base
        # class.
        self.nk = sum(stage_breakdown)


    def setup(self, **variable_args):
        """ Set up the collocation framework """

        self._initialize_dynamic_model()
        self._initialize_polynomial_coefs()
        self._initialize_variables(**variable_args)
        self._initialize_polynomial_constraints()

    def initialize(self, **kwargs):
        """ Call after setting up boundary kinetics, finalizes the
        initialization and sets up the NLP problem. Keyword arguments are
        passed directly as options to the NLP solver """

        self._initialize_solver(**kwargs)

    def plot_optimal(self):
        """ Method to quickly plot an optimal solution. """

        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('darkgrid')
        sns.set_context('talk')

        fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1, figsize=(12,10))

        # Plot the results
        lines = ax[0].plot(self.ts, self.sol[:,1:], '.--')
        ax[0].legend(lines, self.boundary_species[1:], loc='upper left', ncol=2)
    
        # Plot the optimal fluxes
        self._plot_optimal_rates(ax[1])

        # Plot the biomass results
        lines = ax[2].plot(self.ts, self.sol[:,0], '.--')
        ax[2].legend(lines, self.boundary_species, loc='upper left', ncol=2)
        ylim = ax[2].get_ylim()
        ax[2].set_ylim([0, ylim[1]])

        plt.show()


    def add_boundary_constraints(self, constraint_dictionary):
        """ Add dynamic constraints to flux across the system boundary in order
        to account for enzyme kinetics. 
        
        constraint_dictionary should be a dictionary of metabolite : function
        pairs, where 

            `metabolite` is in self.boundary_species, and

            `function` is a function taking a single argument, `x`, a
            dictionary containing symbolic representions of the current
            boundary concentrations, and return a tuple of (lb, ub) specifying
            the symbolic lower and upper bounds of the flux at that finite
            element. A `None` for either lb or ub will skip that constraint,
            using the constant bound specified from the model.

        """

        for key in constraint_dictionary.keys():
            assert key in self.boundary_species, "{} not found".format(key)
    
        self._constraint_fns = constraint_dictionary

        # Iterate over each point
        for k in range(self.nk):
            for j in range(1, self.d+1):
            
                # Create a dictionary to pass to the bounds functions
                x = {met : var_sx for met, var_sx in 
                     zip(self.boundary_species, self.var.x_sx[k,j])}

                for met, boundary_func in constraint_dictionary.items():
                    rxn_sx = self._get_symbolic_flux(k, j)[
                        self.boundary_species.index(met)]
                    lb, ub = boundary_func(x)

                    if lb is not None:
                        self.add_constraint(
                            rxn_sx - lb, 0, cs.inf, 'Boundary constraint lower'
                            'bound - FE {0}, degree {1}'.format(k, j))


                    if ub is not None:
                        self.add_constraint(
                            ub - rxn_sx, 0, cs.inf, 'Boundary constraint upper'
                            'bound - FE {0}, degree {1}'.format(k, j))

    def _initialize_dynamic_model(self):
        """ Initialize the model of biomass growth and substrate accumulation
        """

        t  = cs.SX.sym('t')          # not really used.
        x  = cs.SX.sym('x', self.nx) # External metabolites
        vb = cs.SX.sym('v', self.nx) # fluxes across the boundary

        xdot = vb * x[0]             # Here we assume biomass is in x[0]

        self.dxdt = cs.Function('dxdt', [t,x,vb], [xdot])

    def _initialize_variables(self, pvars=None):

        core_variables = {
            'x'  : (self.nk, self.d+1, self.nx),
            'v'  : (self.nf, self.nv),
            'a'  : (self.nk, self.d),
            'h'  : (self.nf),
        }

        self.var = VariableHandler(core_variables)

        # Initialize default variable bounds
        self.var.x_lb[:] = 0.
        self.var.x_ub[:] = 100.
        self.var.x_in[:] = 1.

        # Initialize EFM bounds. EFMs are nonnegative.
        self.var.v_lb[:] = 0.
        self.var.v_ub[:] = 1.
        self.var.v_in[:] = 0.

        # Activity polynomial.
        self.var.a_lb[:] = 0.
        self.var.a_ub[:] = np.inf
        self.var.a_in[:] = 1.

        # Stage stepsize control (in hours)
        self.var.h_lb[:] = .1
        self.var.h_ub[:] = 10.
        self.var.h_in[:] = 1.

        # Maintain compatibility with codes using a symbolic final time
        self.var.tf_sx = sum([self.var.h_sx[i] * self.stage_breakdown[i]
                              for i in range(self.nf)])

        # We also want the v_sx variable to represent a fraction of the overall
        # efm, so we'll add a constraint saying the sum of the variable must
        # equal 1.
        if self.nf > 1:
            self.add_constraint(cs.sum2(self.var.v_sx[:]), np.ones(self.nf),
                                np.ones(self.nf), 'Sum(v_sx) == 1')
        elif self.nf == 1:
            self.add_constraint(cs.sum1(self.var.v_sx[:]), np.ones(self.nf),
                                np.ones(self.nf), 'Sum(v_sx) == 1')

        if pvars is None: pvars = {}
        self.pvar = VariableHandler(pvars)

    def _get_stage_index(self, finite_element):
        """ Find the current stage based on the indexed finite_element """
        return bisect.bisect(np.cumsum(self.stage_breakdown), finite_element)

    def _get_symbolic_flux(self, finite_element, degree):
        """ Get a symbolic expression for the boundary fluxes at the given
        finite_element and polynomial degree """

        return cs.mtimes(cs.DM(self.efms_object.T.values), 
                         self.var.a_sx[finite_element, degree-1] * 
                         self.var.v_sx[self._get_stage_index(finite_element)])
    
    # cs.SX(self.efms_object.T.dot((
    #         np.asarray(self.var.a_sx[finite_element, degree-1], dtype=object) * 
    #         np.asarray(self.var.v_sx[self._get_stage_index(finite_element)],
    #                    dtype=object)).flatten()).values) 

    def _initialize_polynomial_constraints(self):
        """ Add constraints to the model to account for system dynamics and
        continuity constraints """

        # All collocation time points
        T = np.zeros((self.nk, self.d+1), dtype=object)
        for k in range(self.nk):
            for j in range(self.d+1):
                T[k,j] = (self.var.h_sx[self._get_stage_index(k)] * 
                          (k + self.col_vars['tau_root'][j]))


        # For all finite elements
        for k in range(self.nk):

            # For all collocation points
            for j in range(1, self.d+1):

                # Get an expression for the state derivative at the collocation
                # point
                xp_jk = cs.mtimes(cs.DM(self.col_vars['C'][:,j]).T, self.var.x_sx[k]).T

                # Add collocation equations to the NLP.
                # Boundary fluxes are calculated by multiplying the EFM
                # coefficients in V by the efm matrix
                [fk] = self.dxdt.call(
                    [T[k,j], cs.SX(self.var.x_sx[k,j]),
                     self._get_symbolic_flux(k, j)])

                self.add_constraint(
                    self.var.h_sx[self._get_stage_index(k)] * fk - xp_jk,
                    msg='DXDT collocation - FE {0}, degree {1}'.format(k,j))

            # Add continuity equation to NLP
            if k+1 != self.nk:
                
                # Get an expression for the state at the end of the finite
                # element
                self.add_constraint(
                    cs.SX(self.var.x_sx[k+1,0]) -
                    self._get_endpoint_expr(self.var.x_sx[k]),
                                    msg='Continuity - FE {0}'.format(k))

        # Get an expression for the endpoint for objective purposes
        xf = self._get_endpoint_expr(self.var.x_sx[-1])
        self.xf = {met : x_sx for met, x_sx in zip(self.boundary_species, xf)}

        # Similarly, get an expression for the beginning point
        x0 = self.var.x_sx[0,0,:]
        self.x0 = {met : x_sx for met, x_sx in zip(self.boundary_species, x0)}


    def _plot_optimal_rates(self, ax):

        vs = np.zeros((self.nk, self.d, self.nv))
        for k in range(self.nk):
            for j in range(1, self.d+1):
                vs[k,j-1,:] = (self.var.a_op[k, j-1] *
                               self.var.v_op[self._get_stage_index(k)])

        vs_flat = vs.reshape((self.nk*(self.d)), self.nv)
        vs_sum = vs_flat.sum(0) 
        active_fluxes = vs_sum > 1E-2 * vs_sum.max()

        def build_rxn_string(efm):
            efm = efm.copy()
            efm /= -efm[efm < -1E-6].sum()
            efm.index = efm.index.str.replace('_e', '').str.replace('EX_', '')
            reactants = efm[efm < -1E-2]
            products = efm[efm > 1E-2]
            reactant_bits = ' + '.join(['{:0.2f} {}'.format(-stoich, name) for
                                        name, stoich in reactants.items()])
            product_bits = ' + '.join(['{:0.2f} {}'.format(stoich, name) for
                                       name, stoich in products.items()])
            return reactant_bits + ' --> ' + product_bits
    
        rxn_strings = [build_rxn_string(efm) for name, efm in
                       self.efms_float[active_fluxes].T.items()]

        lines = ax.plot(self.col_vars['tgrid'][:,1:].flatten(), 
                        vs_flat[:, active_fluxes], '.--')

        ax.legend(lines, rxn_strings, loc='upper right')

    def _plot_setup(self):

        self.var.tf_op = sum([self.var.h_op[i] * self.stage_breakdown[i]
                              for i in range(self.nf)])

        stage_starts = np.hstack(
            [np.array(0.), np.cumsum([self.var.h_op[self._get_stage_index(k)]
                                      for k in range(self.nk - 1)])])

        self.col_vars['tgrid'] = np.array(
            [start + self.var.h_op[self._get_stage_index(k)] *
             np.array(self.col_vars['tau_root']) for k, start in
             enumerate(stage_starts)])

        self.ts = self.col_vars['tgrid'].flatten()
        self.sol = self.var.x_op.reshape((self.nk*(self.d+1)), self.nx)


    def _add_relative_stepsize_constraint(self, relative_size=10.):

        for ii in range(self.nf):
            for jj in range(ii+1, self.nf):
                self.add_constraint(self.var.h_sx[ii]/self.var.h_sx[jj],
                                    1./relative_size, relative_size,
                                    msg='relative stepsize constraint')


    def _interpolate_solution(self, ts):

        out = np.empty((len(ts), self.nx))
        stage_starts = [0.]
        for i in range(self.nk-1):
            stage_starts += [self.var.h_op[self._get_stage_index(i)] +
                             stage_starts[-1]]
        stage_starts = pd.Series(stage_starts)
        stages = stage_starts.searchsorted(ts, side='right') - 1

        for ki in range(self.nk):
            for ni in range(self.nx):
                interp = lagrange(self.col_vars['tau_root'], 
                                  self.var.x_op[ki, :, ni])

                out[stages == ki, ni] = interp(
                    (ts[stages == ki] - stage_starts[ki]) /
                    self.var.h_op[self._get_stage_index(ki)])

        return out

    def _interpolate_derivative(self, ts):

        out = np.empty((len(ts), self.nx))
        stage_starts = [0.]
        for i in range(self.nk-1):
            stage_starts += [self.var.h_op[self._get_stage_index(i)] +
                             stage_starts[-1]]
        stage_starts = pd.Series(stage_starts)
        stages = stage_starts.searchsorted(ts, side='right') - 1

        for ki in range(self.nk):
            for ni in range(self.nx):
                interp = lagrange(self.col_vars['tau_root'], 
                                  (self.col_vars['C'].T.dot(
                                      self.var.x_op[ki, :, ni]) /
                                   self.var.h_op[self._get_stage_index(ki)]))

                out[stages == ki, ni] = interp(
                    (ts[stages == ki] - stage_starts[ki]) /
                    self.var.h_op[self._get_stage_index(ki)])

        return out

    # def _interpolate_boundary_constraints(self, ts):
    #
    #     stage_starts = [0.]
    #     for i in range(self.nk-1):
    #         stage_starts += [self.var.h_op[self._get_stage_index(i)] +
    #                          stage_starts[-1]]
    #     stage_starts = pd.Series(stage_starts)
    #     stages = stage_starts.searchsorted(ts, side='right') - 1
    #
    #     for ki in range(self.nk):
    #         for ji in range(1, self.d+1):
    #
    #             x = {met : var_op for met, var_op in 
    #                  zip(self.boundary_species, self.var.x_op[k,j])}
    #
    #             interp = lagrange(self.col_vars['tau_root'], 
    #                               (self.col_vars['C'].T.dot(
    #                                   self.var.x_op[ki, :, ni]) /
    #                                self.var.h_op[self._get_stage_index(ki)]))
    #
    #             out[stages == ki, ni] = interp(
    #                 (ts[stages == ki] - stage_starts[ki]) /
    #                 self.var.h_op[self._get_stage_index(ki)])
    #
    #     return out
