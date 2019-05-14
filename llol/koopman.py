'''
A generic interface for koopman-based approximations.

Stanley Bak
March 2019
'''

import matplotlib.pyplot as plt
import numpy as np

from util import Freezable, Timers
from settings import Settings

class Koopman(Freezable):
    'generic koopman container'

    def __init__(self, settings):
        assert isinstance(settings, Settings)

        self.settings = settings

        self.a_mat = None # created by make_approx
        self.eobs_func = None # created by make_approx
        
        self.freeze_attrs()

    def make_approx(self):
        'make the linear approximation according to the settings'

        Timers.tic('make_approx')

        self.eobs_func = self.make_eobs_func()

        x_mat, y_mat = self.settings.data_source.make_data(self.settings.der_func, self.eobs_func)

        # do regression
        self.do_regression(x_mat, y_mat)

        Timers.toc('make_approx')

    def make_eobs_func(self):
        'make a function that converts a matrix of states to a matrix of (extended) observations'

        s = self.settings
        n = s.dims

        output_dims = n if s.include_original_vars else 0

        # power basis
        if s.power_order is not None:
            output_dims += s.power_order ** n

        # trig basis

        # hermite basis

        if output_dims == 0:
            s.include_original_vars = True
            output_dims = n

        print(f"Extended Observation Length: {n}")

        def eobs_func(state_mat):
            'returns a matrix of extended observations based on the passed-in state matrix'

            # work with rows because it's faster
            state_mat = state_mat.T.copy()

            Timers.tic('eobs_func')

            eobs_mat = np.zeros((state_mat.shape[1], output_dims), dtype=float)

            print(f"eobs shape: {eobs_mat.shape}")

            for state, eobs in zip(state_mat, eobs_mat):
                index = 0

                # original
                if s.include_original_vars:
                    for i in range(n):
                        eobs[index] = state[i]
                        index += 1

                # power basis
                if s.power_order is not None:
                    for iterator in s.power_order ** n:
                        val = 1

                        temp = iterator
                        
                        # extract the iterator for each dimension
                        for dim_num in range(n):
                            deg = temp % s.power_order
                            temp = temp // s.power_order

                            val *= state[dim_num]**deg

                        eobs[index] = val
                        index += 1

            Timers.toc('eobs_func')
                
            return eobs_mat.transpose().copy()

        return eobs_func

    def do_regression(self, x1, x2):
        '''do regression on the x and y matrices

        This produces self.a_mat
        '''

        Timers.tic('regression')
        
        if self.settings.pseudo_inverse_method == Settings.DIRECT:
            Timers.tic('pinv')
            x_pseudo = np.linalg.pinv(x1)
            Timers.toc('pinv')
            Timers.tic('dot')
            self.a_mat = np.dot(x2, x_pseudo)
            Timers.toc('dot')

            print(f"after regression, x1 shape: {x1.shape}, x2 shape: {x2.shape}, a_mat shape: {self.a_mat.shape}")
        else:
            raise RuntimeError(f"Unimplemented pesudo-inverse method: {self.pseudo_inverse_methods}")

        Timers.toc('regression')

    def plot_approx(self, init, npoints, xdim=0, ydim=1, col='k-', label='eDMD Approx', max_norm=float('inf')):
        'plot the approximation'

        Timers.tic('plot_approx')

        xs = []
        ys = []

        init = np.array(init, dtype=float)
        init.shape = (self.settings.dims, 1)

        estate = self.eobs_func(init)

        xs.append(estate[xdim])
        ys.append(estate[ydim])

        print(f"estate shape: {estate.shape}, a_mat shape: {self.a_mat.shape}")

        for step in range(npoints):
            Timers.tic('dot')
            estate = np.dot(self.a_mat, estate)
            Timers.toc('dot')

            x = estate[xdim]
            y = estate[ydim]

            if np.linalg.norm([x, y]) > max_norm:
                print(f"Approximation was bad at step {step} (plotting stopped prematurely)")
                break

            xs.append(x)
            ys.append(y)

        plt.plot(xs, ys, col, label=label)

        Timers.toc('plot_approx')

    def save_plot(self, filename):
        'save the plot to a file'

        plt.legend()
        plt.savefig('vanderpol.png')
