'''
Settings for generic koopman interface.

Stanley Bak
March, 2019
'''

from util import Freezable

class Settings(Freezable):
    'Koopman approximation settings'

    DIRECT, = range(1) # pseudo-inverse method

    def __init__(self):

        # required to be set
        self.dims = None
        self.der_func = None

        self.data_source = None

        self.pseudo_inverse_method = Settings.DIRECT

        # settings related to extended observations
        self.include_original_vars = True
        self.power_order = None # power basis maximum order. order 2 has: x^2*y^2
        self.trig_order = None # trig basis maximum order. order 2 has: sin(2x)
        self.hermite_order = None # hermite basis maximum order. order 2 has: (4x^2-2)*(4y^2-2)
        

        self.freeze_attrs()

    def check_settings(self):
        'check that required settings were assigned'

        assert self.dims is not None
        assert self.der_func is not None

        self.data_source.check_settings()
