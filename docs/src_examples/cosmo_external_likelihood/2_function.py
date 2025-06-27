import matplotlib.pyplot as plt
import numpy as np

_do_plot = False


def my_like(
    # Parameters that we may sample over (or not)
    noise_std_pixel=20,  # muK
    beam_FWHM=0.25,  # deg
    # Keyword through which the cobaya likelihood instance will be passed.
    _self=None,
):
    # Noise spectrum, beam-corrected
    healpix_Nside = 512
    pixel_area_rad = np.pi / (3 * healpix_Nside**2)
    weight_per_solid_angle = (noise_std_pixel**2 * pixel_area_rad) ** -1
    beam_sigma_rad = beam_FWHM / np.sqrt(8 * np.log(2)) * np.pi / 180.0
    ells = np.arange(l_max + 1)
    Nl = np.exp((ells * beam_sigma_rad) ** 2) / weight_per_solid_angle
    # Cl of the map: data + noise
    Cl_map = Cl_est + Nl
    # Request the Cl from the provider
    Cl_theo = _self.provider.get_Cl(ell_factor=False, units="muK2")["tt"][: l_max + 1]
    Cl_map_theo = Cl_theo + Nl
    # Auxiliary plot
    if _do_plot:
        ell_factor = ells * (ells + 1) / (2 * np.pi)
        plt.figure()
        plt.plot(ells[2:], (Cl_theo * ell_factor)[2:], label=r"Theory $C_\ell$")
        plt.plot(
            ells[2:], (Cl_est * ell_factor)[2:], label=r"Estimated $C_\ell$", ls="--"
        )
        plt.plot(ells[2:], (Cl_map * ell_factor)[2:], label=r"Map $C_\ell$")
        plt.plot(ells[2:], (Nl * ell_factor)[2:], label="Noise")
        plt.legend()
        plt.ylim([0, 6000])
        plt.savefig(_plot_name)
        plt.close()
    # ----------------
    # Compute the log-likelihood
    V = Cl_map[2:] / Cl_map_theo[2:]
    logp = np.sum((2 * ells[2:] + 1) * (-V / 2 + 1 / 2.0 * np.log(V)))
    # Set our derived parameter
    derived = {"Map_Cl_at_500": Cl_map[500]}
    return logp, derived
