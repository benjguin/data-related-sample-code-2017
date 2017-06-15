from __future__ import print_function
import numpy as np
import collections
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.learners import momentum_sgd, fsadagrad, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk import input, cross_entropy_with_softmax, classification_error, sequence, \
                element_select, alias, hardmax, placeholder_variable, combine, parameter, times, plus
from cntk.ops.functions import CloneMethod, load_model, Function
from cntk.initializer import glorot_uniform
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import plot
from cntk.layers import *
from cntk.layers.sequence import *
from cntk.layers.models.attention import *
from cntk.layers.typing import *

# variables you may want to change
training_file_path = "./data/text1.ctf"

# configure CNTK to use CPU or GPU
vm_config="cpu" # default value
if 'TEST_DEVICE' in os.environ:
    vm_config = os.environ['TEST_DEVICE']
else:
    print("TEST_DEVICE environment variable was not found, using %s as a default value"%vm_config)

print("vm_config=%s"%vm_config)

# Select the right target device when this notebook is being tested:
if vm_config == "gpu":
    import cntk
    print("using GPU")
    cntk.device.set_default_device(cntk.device.gpu(0))
else:
    import cntk
    print("using CPU")
    cntk.device.try_set_default_device(cntk.device.cpu())

# Because of randomization in training we set a fixed random seed to ensure repeatable outputs
from _cntk_py import set_fixed_random_seed
set_fixed_random_seed(34)

