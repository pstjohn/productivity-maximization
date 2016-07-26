import pandas as pd
import numpy as np

import casadi as cs

from functools import reduce
from operator import mul
import itertools
from collections import Iterable

class VariableHandler(object):

    def __init__(self, shape_dict):
        """ A class to handle the flattening and expanding of the NLP variable
        vector. Solves a lot of headaches.

        shape_dict: a dictionary containing variable_name : shape pairs, where
        shape is a tuple of the desired variable dimensions.
        
        """

        self._data = pd.DataFrame(pd.Series(shape_dict), columns=['shapes'])

        self._data['lengths'] = self._data.shapes.apply(product)
        self._data['end'] = self._data.lengths.cumsum()
        self._data['start'] = self._data.end - self._data.lengths
        self._total_length = self._data.lengths.sum()

        # Initialize symbolic variable
        self.vars_sx = cs.SX.sym('vars', self._total_length)

        # Split symbolic variable
        symbolic_dict = self._expand(self.vars_sx)

        for key, row in self._data.iterrows():
            self.__dict__.update({
                key + '_lb' : np.zeros(row.shapes), # Lower bounds
                key + '_ub' : np.zeros(row.shapes), # Upper bounds
                key + '_in' : np.zeros(row.shapes), # Initial guess
                key + '_op' : np.zeros(row.shapes), # Optimized Value
                key + '_sx' : symbolic_dict[key],
           })


    @property
    def vars_lb(self): return self._condense('lb')

    @vars_lb.setter
    def vars_lb(self, vars_lb):
        expanded = self._expand(vars_lb)
        for key, val in expanded.items():
            self.__dict__.update({key + '_lb' : val})


    @property
    def vars_ub(self): return self._condense('ub')

    @vars_ub.setter
    def vars_ub(self, vars_ub):
        expanded = self._expand(vars_ub)
        for key, val in expanded.items():
            self.__dict__.update({key + '_ub' : val})


    @property
    def vars_in(self): return self._condense('in')

    @vars_in.setter
    def vars_in(self, vars_in):
        expanded = self._expand(vars_in)
        for key, val in expanded.items():
            self.__dict__.update({key + '_in' : val})


    @property
    def vars_op(self): return self._condense('op')

    @vars_op.setter
    def vars_op(self, vars_op):
        expanded = self._expand(vars_op)
        for key, val in expanded.items():
            self.__dict__.update({key + '_op' : val})


    def _condense(self, suffix):
        """ Flatten the given variables to give a single dimensional vector """

        vector_out = np.zeros(self._total_length)
        for key, row in self._data.iterrows():
            try:
                vector_out[row.start:row.end] = \
                    self.__dict__[key + '_' + suffix].flatten()
            except AttributeError:
                # Allow for pandas dataframes
                vector_out[row.start:row.end] = \
                    self.__dict__[key + '_' + suffix].values.flatten()

        return vector_out


    def _expand(self, vector):
        """ Given a flattened vector, expand into the component matricies """

        def reshape_slice(row, key):
            if isinstance(vector, cs.SX):
                return LinearSlicer(vector[row.start:row.end], row.shapes)
            else:
                return vector[row.start:row.end].reshape(row.shapes)

        return {key : reshape_slice(row, key) for key, row in
                self._data.iterrows()}

        # return pd.Series([reshape_slice(row, key) for key, row in
        #                   self._data.iterrows()], index = self._data.index)

    def __getstate__(self):
        result = self.__dict__.copy()
        to_delete = [key for key in result.keys() if key.endswith('_sx')]
        for key in to_delete:
            del result[key]
        return result
    
    def __setstate__(self, result):
        self.__dict__ = result
        self.vars_sx = cs.SX.sym('vars', self._total_length)
        symbolic_dict = self._expand(self.vars_sx)
        for key, row in self._data.iterrows():
            self.__dict__.update({
                key + '_sx' : symbolic_dict[key],
            }) 


class LinearSlicer(object):
    def __init__(self, data, shape):
        """Class to handle the slicing of an object stored as a C-continuous
        casadi SX list """
        
        if not isinstance(shape, Iterable):
            shape = [shape,]
        
        assert data.size1() == product(shape)
        
        self._data = data
        self._shape = shape
        self._ndim = len(shape)
    
    def __getitem__(self, indices):
        
        if not isinstance(indices, Iterable):
            indices = [indices,]
        else:
            indices = list(indices)
            
        if len(indices) < self._ndim:
            indices += [slice(None, None, None)]*(
                self._ndim - len(indices))
            
        for i, index in enumerate(indices):
            if isinstance(index, slice):
                indices[i] = list(range(*index.indices(self._shape[i])))

            elif isinstance(index, int):
                if index < 0 : #Handle negative indices
                    index += self._shape[i]
                indices[i] = [index,]
            
        desired_shape = [len(i) for i in indices if len(i) is not 1]
        if len(desired_shape) < 2:
            desired_shape += [1]*(2 - len(desired_shape))
        elif len(desired_shape) > 2:
            raise AssertionError('Casadi matrices can only be 2-D')

        multi_index = np.array(list(itertools.product(*indices)))
        multi_index_tuples = tuple([elem for elem in multi_index.T])
        
        raveled_index = np.ravel_multi_index(multi_index_tuples, self._shape)
        
        return self._data[raveled_index].reshape(
            (desired_shape[1], desired_shape[0])).T
    

def product(iterable):
    try: return reduce(mul, iterable)
    except TypeError: return iterable
