'''
Data sources for Koopman DMD

Stanley Bak
March 2019
'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from util import Freezable, Timers

class SingleSimulationData(Freezable):
    'Get pair-data from a single simulation'

    def __init__(self):
        self.tmax = None
        self.npoints = None
        self.init = None

        # used if plot is True
        self.plot = False
        self.plot_xdim = 0
        self.plot_ydim = 1
        self.plot_color = 'r+'
        self.plot_lw = 1
        self.plot_label = 'Simulation Data'

        self.freeze_attrs()

    def check_settings(self):
        'check that required settings were set'
        
        assert self.tmax is not None
        assert self.npoints is not None
        assert self.init is not None

    def make_data(self, der_func, eobs_func):
        '''generate the data for regression

        der_func is a derivative function given a single state
        eobs_func converts a matrix of states to a matrix of (extended) observations

        returns (x_mat, y_mat) where the regression problem is y_mat = A * x_mat

        each column of x_mat / y_mat is one set of state vectors
        '''

        Timers.tic('make_data')

        n = len(self.init)

        times = np.linspace(0, self.tmax, self.npoints)

        def der(state, _):
            'derivative function (reversed args for odeint)'

            return der_func(state)

        Timers.tic('odeint')
        sol = odeint(der, self.init, times).T
        Timers.toc('odeint')

        assert sol.shape[0] == n
        assert sol.shape[1] == self.npoints

        if self.plot:
            Timers.tic('plot')
            xs = [data[self.plot_xdim] for data in sol.T]
            ys = [data[self.plot_ydim] for data in sol.T]

            plt.plot(xs, ys, self.plot_color, label=self.plot_label)
            Timers.toc('plot')

        # extend observations
        mat = eobs_func(sol)

        x_mat = mat[:-1]
        y_mat = mat[1:]

        Timers.toc('make_data')

        return x_mat, y_mat
