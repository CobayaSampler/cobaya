from .gridconfig import *
from .runbatch import *
from . import jobqueue

jobqueue.set_default_program('cobaya-run', 'COBAYA')
