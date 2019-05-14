'''
Vanderpol example using generic koopman interface

Stanley Bak
March 2019
'''

from koopman import Koopman
from settings import Settings
from data_source import SingleSimulationData
from util import Timers

def der_func(state):
    'derivative function'

    x, y = state

    dx = y
    dy = y - x - x**2*y

    return [dx, dy]

def main():
    'main entry point'

    Timers.tic('total')

    sim_data = SingleSimulationData()
    sim_data.tmax = 6.5
    sim_data.npoints = 100
    sim_data.init = [1.4, 2.4]
    sim_data.plot = True

    s = Settings()
    s.der_func = der_func
    s.dims = 2
    s.data_source = sim_data

    koop = Koopman(s)

    koop.make_approx()

    koop.plot_approx(sim_data.init, sim_data.npoints, max_norm=10)

    koop.save_plot('vanderpol.png')

    Timers.toc('total')
    Timers.print_stats()

if __name__ == '__main__':
    main()
