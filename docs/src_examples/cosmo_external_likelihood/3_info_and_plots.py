info = {
    'params': {
        # Fixed
        'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
        'mnu': 0.06, 'nnu': 3.046,
        # Sampled
        'As': {'prior': {'min': 1e-9, 'max': 4e-9}, 'latex': 'A_s'},
        'ns': {'prior': {'min': 0.9, 'max': 1.1}, 'latex': 'n_s'},
        'noise_std_pixel': {
            'prior': {'dist': 'norm', 'loc': 20, 'scale': 5},
            'latex': r'\sigma_\mathrm{pix}'},
        # Derived
        'Map_Cl_at_500': {'latex': r'C_{500,\,\mathrm{map}}'}},
    'likelihood': {'my_cl_like': {
        "external": my_like,
        # Declare required quantities!
        "requires": {'Cl': {'tt': l_max}},
        # Declare derived parameters!
        "output_params": ['Map_Cl_at_500']}},
    'theory': {'camb': {'stop_at_error': True}},
    'packages_path': packages_path}

from cobaya.model import get_model
model = get_model(info)

# Eval likelihood once with fid values and plot
_do_plot = True
_plot_name = "fiducial.png"
fiducial_params_w_noise = fiducial_params.copy()
fiducial_params_w_noise['noise_std_pixel'] = 20
model.logposterior(fiducial_params_w_noise)
_do_plot = False

# Plot of (prpto) probability density
As = np.linspace(1e-9, 4e-9, 10)
loglikes = [model.loglike({'As': A, 'ns': 0.96, 'noise_std_pixel': 20})[0] for A in As]
plt.figure()
plt.plot(As, loglikes)
plt.gca().get_yaxis().set_visible(False)
plt.title(r"$\log P(A_s|\mathcal{D},\mathcal{M}) (+ \mathrm{const})$")
plt.xlabel(r"$A_s$")
plt.savefig("log_like.png")
plt.close()
