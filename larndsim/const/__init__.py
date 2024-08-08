from . import detector, light
import numpy as np

def load_properties(detprop_file, pixel_file):
    """
    The function loads the detector properties,
    the pixel geometry, and the simulation YAML files
    and stores the constants as global variables

    Args:
        detprop_file (str): detector properties YAML filename
        pixel_file (str): pixel layout YAML filename
        sim_file (str): simulation properties YAML filename
    """
    detector.set_detector_properties(detprop_file, pixel_file)
    light.set_light_properties(detprop_file)

import os

print('Setting light properties')
detprop_file = os.path.join(os.path.dirname(__file__), 'module0.yml')

light.set_light_properties(os.path.join(os.path.dirname(__file__), 'module0.yml'))
detector.MODULE_TO_TPCS = detector.get_n_modules(detprop_file)
detector.TPC_TO_MODULE = np.arange(48)
