import numpy as np
import pandas as pd
import casadi as cs


class BaseCollocation(object):

    def __init__(self):

        # Set up defaults for these collocation parameters (can be re-assigned
        # prior to collocation initialization
        self.nk = 20
        self.d = 2
        
        # Initialize container variables
        self.col_vars = {}
        self._constraints_sx = []
        self._constraints_lb = []
        self._constraints_ub = []
        self.objective_sx = 0.

    def add_constraint(self, sx, lb=None, ub=None, msg=None):
        """ Add a constraint to the problem. sx should be a casadi symbolic
        variable, lb and ub should be the same length. If not given, upper and
        lower bounds default to 0. 

        msg (str):
            A description of the constraint, to be raised if it throws an nan
            error

        Replaces manual addition of constraint variables to allow for warnings
        to be issues when a constraint that returns 'nan' with the current
        initalized variables is added.

        """

        constraint_len = sx.shape[0]
        assert sx.shape[1] == 1, "SX shape {} mismatch".format(sx.shape)

        if lb is None: lb = np.zeros(constraint_len)
        else: lb = np.atleast_1d(np.asarray(lb))

        if ub is None: ub = np.zeros(constraint_len)
        else: ub = np.atleast_1d(np.asarray(ub))

        # Make sure the bounds are sensible
        assert len(lb) == constraint_len, "LB length mismatch"
        assert len(ub) == constraint_len, "UB length mismatch"
        assert np.all(lb <= ub), "LB ! <= UB"

        try:
            gfcn = cs.Function('g_test',
                                 [self.var.vars_sx, self.pvar.vars_sx],
                                 [sx])
            out = np.asarray(gfcn(self.var.vars_in, self.pvar.vars_in))
            if np.any(np.isnan(out)):
                error_states = np.array(self.boundary_species)[
                    np.where(np.isnan(out))[0]]
                raise RuntimeWarning('Constraint yields NAN with given input '
                                     'arguments: \nConstraint:\n\t{0}\n'
                                     'Offending states: {1}'.format(
                                         msg, error_states))
        
        except (AttributeError, KeyError):
            pass

        self._constraints_sx.append(sx)
        self._constraints_lb.append(lb)
        self._constraints_ub.append(ub)



    def solve(self):
        """ Solve the NLP. Alpha specifies the value for the regularization
        parameter, which minimizes the sum |v|.

        """
        
        # Fill argument call dictionary
        arg = {
            'x0'  : self.var.vars_in,
            'lbx' : self.var.vars_lb,
            'ubx' : self.var.vars_ub,

            'lbg' : self.col_vars['lbg'],
            'ubg' : self.col_vars['ubg'],

            'p'   : self.pvar.vars_in,
        }


        # Call the solver
        self._result = self._solver.call(arg)

        if self._solver.stats()['return_status'] not in [
                'Solve_Succeeded', 'Solved_To_Acceptable_Level']:
            raise RuntimeWarning('Solve status: {}'.format(
                self._solver._solver.stats()['return_status']))

        # Process the optimal vector
        self.var.vars_op = np.asarray(self._result['x'])

        # Store the optimal solution as initial vectors for the next go-around
        self.var.vars_in = np.asarray(self.var.vars_op)

        try: self._plot_setup()
        except AttributeError: pass

        return float(self._result['f'])


    def _initialize_polynomial_coefs(self):
        """ Setup radau polynomials and initialize the weight factor matricies
        """
        self.col_vars['tau_root'] = [0] + cs.collocation_points(self.d, "radau")

        # Dimensionless time inside one control interval
        tau = cs.SX.sym("tau")

        # For all collocation points
        L = [[]]*(self.d+1)
        for j in range(self.d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the
            # collocation point
            L[j] = 1
            for r in range(self.d+1):
                if r != j:
                    L[j] *= (
                        (tau - self.col_vars['tau_root'][r]) / 
                        (self.col_vars['tau_root'][j] -
                         self.col_vars['tau_root'][r]))

        self.col_vars['lfcn'] = lfcn = cs.Function(
            'lfcn', [tau], [cs.vertcat(*L)])

        # Evaluate the polynomial at the final time to get the coefficients of
        # the continuity equation
        # Coefficients of the continuity equation
        self.col_vars['D'] = np.asarray(lfcn(1.0)).squeeze()

        # Evaluate the time derivative of the polynomial at all collocation
        # points to get the coefficients of the continuity equation
        tfcn = lfcn.tangent()

        # Coefficients of the collocation equation
        self.col_vars['C'] = np.zeros((self.d+1, self.d+1))
        for r in range(self.d+1):
            self.col_vars['C'][:,r] = np.asarray(tfcn(self.col_vars['tau_root'][r])[0]).squeeze()

        # Find weights for gaussian quadrature: approximate int_0^1 f(x) by
        # Sum(
        xtau = cs.SX.sym("xtau")

        Phi = [[]] * (self.d+1)

        for j in range(self.d+1):
            dae = dict(t=tau, x=xtau, ode=L[j])
            tau_integrator = cs.integrator(
                "integrator", "cvodes", dae, {'t0':0., 'tf':1})
            Phi[j] = np.asarray(tau_integrator(x0=0)['xf'])

        self.col_vars['Phi'] = np.array(Phi)
        
    def _initialize_solver(self, **kwargs):

        nlpsol_args = {"expand", "iteration_callback",
                       "iteration_callback_step",
                       "iteration_callback_ignore_errors", "ignore_check_vec",
                       "warn_initial_bounds", "eval_errors_fatal",
                       "print_time", "verbose_init"}

        # Initialize NLP object
        opts = {
            'ipopt.max_iter' : 10000,
            # 'linear_solver' : 'ma27'
        }
        
        if kwargs is not None: 
            for key, val in kwargs.items(): 
                if key in nlpsol_args:
                    opts.update({key: val })
                else:
                    opts.update({'ipopt.' + key: val })

        self._solver_opts = opts
        constraints = cs.vertcat(*self._constraints_sx)




        self._solver = cs.nlpsol(
            "solver", "ipopt",
            {'x': self.var.vars_sx,
             'p': self.pvar.vars_sx,
             'f': self.objective_sx,
             'g': constraints},
            self._solver_opts)

        self.col_vars['lbg'] = np.concatenate(self._constraints_lb)
        self.col_vars['ubg'] = np.concatenate(self._constraints_ub)

    # def warm_solve(self, x0=None, lam_x=None, lam_g=None):
    #     """Solve the collocation problem using an initial guess and basis from
    #     a prior solve. Defaults to using the variables from the solve stored in
    #     _results. 
    #
    #     """
    #     warm_solve_opts = dict(self._solver_opts)
    #
    #     warm_solve_opts["warm_start_init_point"] = "yes"
    #     warm_solve_opts["warm_start_bound_push"] = 1e-6
    #     warm_solve_opts["warm_start_slack_bound_push"] = 1e-6
    #     warm_solve_opts["warm_start_mult_bound_push"] = 1e-6
    #
    #     solver = self._solver = cs.NlpSolver("solver", "ipopt", self._nlp,
    #                                          warm_solve_opts)
    #
    #
    #     if x0 is None: x0 = self._result['x']
    #     if lam_x is None: lam_x = self._result['lam_x']
    #     if lam_g is None: lam_g = self._result['lam_g']
    #
    #     solver.setInput(x0, 'x0')
    #     solver.setInput(self.var.vars_lb, 'lbx')
    #     solver.setInput(self.var.vars_ub, 'ubx')
    #     solver.setInput(self.col_vars['lbg'], 'lbg')
    #     solver.setInput(self.col_vars['ubg'], 'ubg')
    #     solver.setInput(self.pvar.vars_in, 'p')
    #     solver.setInput(lam_x, 'lam_x0')
    #     solver.setInput(lam_g, 'lam_g0')
    #     solver.setOutput(lam_x, "lam_x")
    #
    #     self._solver.evaluate()
    #
    #     if self._solver.getStat('return_status') != 'Solve_Succeeded':
    #         raise RuntimeWarning('Solve status: {}'.format(
    #             self._solver.getStat('return_status')))
    #
    #     self._result = {
    #         'x' : self._solver.getOutput('x'),
    #         'lam_x' : self._solver.getOutput('lam_x'),
    #         'lam_g' : self._solver.getOutput('lam_g'),
    #         'f' : self._solver.getOutput('f'),
    #     }
    #
    #     # Process the optimal vector
    #     self.var.vars_op = self._result['x']
    #
    #     # Store the optimal solution as initial vectors for the next go-around
    #     self.var.vars_in = self.var.vars_op
    #
    #     try: self._plot_setup()
    #     except AttributeError: pass
    #
    #     return float(self._result['f'])

    def _get_endpoint_expr(self, state_sx):
        """Use the variables in self.col_vars['D'] for find an expression for
        the end of the finite element """
        return cs.mtimes(cs.DM(self.col_vars['D']).T, state_sx).T
        

    def __getstate__(self):
        result = self.__dict__.copy()
        result['col_vars'] = self.__dict__['col_vars'].copy()

        del result['_constraints_sx']
        del result['_constraints_lb']
        del result['_constraints_ub']
        del result['objective_sx']
        del result['col_vars']['lfcn']
        del result['col_vars']['lbg']
        del result['col_vars']['ubg']
        del result['dxdt']
        del result['_solver']
        del result['_nlp']

        del result['x0']
        del result['xf']

        return result

    def __setstate__(self, result):
        self.__dict__ = result
        self._initialize_polynomial_coefs()



@cs.pycallback
class IterationCallback(object):
    def __init__(self):
        """ A class to store intermediate optimization results. Should be
        passed as an initialized object to ```initialize_solver``` under the
        keyword "iteration_callback". """

        self.iteration = 0
        self._x_data = {}
        self._f_data = {}

    def __call__(self, f, *args):
        self.iteration += 1

        self._x_data[self.iteration] = np.asarray(f.getOutput('x')).flatten()
        self._f_data[self.iteration] = float(f.getOutput('f'))

    @property
    def x_data(self):
        return pd.DataFrame(self._x_data)

    @property
    def f_data(self):
        return pd.Series(self._f_data)



