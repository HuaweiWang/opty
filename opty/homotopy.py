# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:01:23 2016

@author: huawei
"""

from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d
import sympy as sym
import sympy.physics.mechanics as me
from pydy.codegen.ode_function_generators import generate_ode_function

sym_kwargs = {'positive': True, 'real': True}
me.dynamicsymbols._t = sy.symbols('t', **sym_kwargs)

class HomotopyTransfer():
    
    '''This class transfer input orginal dynamic model into homotopy dynamics.
    
    model_dynamics : sympy.Matrix, shape(n, 1)
            A column matrix of SymPy expressions defining the right hand
            side of the equations of motion when the left hand side is zero,
            e.g. 0 = x'(t) - f(x(t), u(t), p) or 0 = f(x'(t), x(t), u(t),
            p). These should be in first order form but not necessairly
            explicit.
    model_states : iterable
            An iterable containing all of the SymPy functions of time which
            represent the states in the equations of motion. 
    model_sepcified: iterable
    model_par_map : dictionary, optional
            A dictionary that maps the SymPy symbols representing the known
            constant parameters to floats. Any parameters in the equations
            of motion not provided in this dictionary will become free
            optimization variables.
    homotopy_control: float, optional
	       A parameter in homotopy method that controls the change of 
	       motion equations. The default value of it is 0, which means 
             the homotopy method does not apply. 
    tracing_dynamic_control: float, optional
	       A parameter in homotopy method that adjust the dynamics of extra
	       'data tracing' term. The default value of it is 1.  
    '''
    
    def __init__(self, model_dynamics, model_states, model_sepcified, model_par_map, homotopy_control=0, tracing_dynamic_control=10):
        
        self.model_dynamics = model_dynamics
        
        self.state_symbols = tuple(state_symbols)
        self.state_derivative_symbols = tuple([s.diff(self.time_symbol) for
                                               s in state_symbols])
                                               
        self.num_states = len(self.state_symbols)
        
        self.specified = model_sepcified
        
        self.model_par_map = model_par_map
        
        self.lamda = homotopy_control
        self.dynamic = tracing_dynamic_control
        
    def _load_states(self):

        self.time = me.dynamicsymbols._t

        syms = 'theta_a, theta_h, omega_a, omega_h'
        time_varying = [s(self.time) for s in
                        sy.symbols(syms, cls=sy.Function, real=True)]

        self.coordinates = OrderedDict()
        self.coordinates['ankle_angle'] = time_varying[0]
        self.coordinates['hip_angle'] = time_varying[1]

        self.speeds = OrderedDict()
        self.speeds['ankle_rate'] = time_varying[2]
        self.speeds['hip_rate'] = time_varying[3]
    
    def _add_sepcified(self):
        
        self.time = me.dynamicsymbols._t
        
        time_varying = [s(self.time)
                        for s in sy.symbols('theta_a_T, theta_h_T, omega_a_T, omega_h_T' ,
                                            cls=sy.Function, real=True)]
        
        self.specified['ankle_angle_tracing'] = time_varying[0]
        self.specified['hip_angle_tracing'] = time_varying[1]
        self.specified['ankle_rate_tracing'] = time_varying[2]
        self.specified['hip_rate_tracing'] = time_varying[3]
        
    def _add_parameter(self):
        
        self.parameters = OrderedDict()
        self.parameters['homotopy_control'] = sy.symbols('l_A', **sym_kwargs)
        self.parameters['tracing_dynamic_control'] = sy.symbols('k_T', **sym_kwargs)        
        
    def _numberical_par(self):
        
        p = {'homotopy_control': self.lamda,
             'tracing_dynamic_control': self.dynamic}
             
        self.open_loop_par_map = OrderedDict()

        for k, v in self.parameters.items():
            self.model_par_map[v] = p[k]             
            
    def _generate_matrix(self):
        
        
        self.Lamda_matrix = sym.zeros(self.num_states)
        self.Lamda_matrix[self.num_states/2:, self.num_states/2:] = self.parameters['homotopy_control']*sym.eye(self.num_states/2)
        
        self.Inv_Lamda_matrix = sym.eye(self.num_states)
        self.Inv_Lamda_matrix[self.num_states/2:, self.num_states/2:] = (1-self.parameters['homotopy_control'])*sym.eye(self.num_states/2)
        
        self.k_matrix = self.parameters['tracing_dynamic_control'] * sym.eye(self.num_states)
        
        
    def _dynamic_transfer(self):
        
        self.homotopy_dynamic = self.Inv_Lamda_matrix*self.model_dynamics + self.Lamda_matrix*(self.state_derivative_symbols - self.k_matrix
		            * sym.matrix([[self.specified['ankle_angle_tracing'] - self.coordinates['ankle_angle']],
                                     [self.specified['hip_angle_tracing'] - self.coordinates['hip_angle']],
                                     [self.specified['ankle_rate_tracing'] - self.speeds['ankle_rate']],
                                     [self.specified['hip_rate_tracing'] - self.speeds['hip_rate']] ])