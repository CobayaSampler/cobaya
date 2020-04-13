fiducial_params = {
    'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
    'As': 2.2e-9, 'ns': 0.96,
    'mnu': 0.06, 'nnu': 3.046}

l_max = 1000

packages_path = '/path/to/your/packages'

info_fiducial = {
    'params': fiducial_params,
    'likelihood': {'one': None},
    'theory': {'camb': {"extra_args": {"num_massive_neutrinos": 1}}},
    'packages_path': packages_path}

from cobaya.model import get_model
model_fiducial = get_model(info_fiducial)

# Declare our desired theory product
# (there is no cosmological likelihood doing it for us)
model_fiducial.add_requirements({"Cl": {'tt': l_max}})

# Compute and extract the CMB power spectrum
# (In muK^-2, without l(l+1)/(2pi) factor)
# notice the empty dictionary below: all parameters are fixed
model_fiducial.logposterior({})
Cls = model_fiducial.provider.get_Cl(ell_factor=False, units="muK2")

# Our fiducial power spectrum
Cl_est = Cls['tt'][:l_max + 1]
