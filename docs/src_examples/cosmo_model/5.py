# Request H(z)
import numpy as np

redshifts = np.linspace(0, 2.5, 40)

theory = model.theory["classy"]
theory.must_provide(Hubble={"z": redshifts})

omega_cdm = [0.10, 0.11, 0.12, 0.13, 0.14]

f, (ax_cl, ax_h) = plt.subplots(1, 2, figsize=(14, 6))
for o in omega_cdm:
    point["omega_cdm"] = o
    model.logposterior(point)  # to force computation of theory
    Cls = theory.get_Cl(ell_factor=True)
    ax_cl.plot(Cls["ell"][2:], Cls["tt"][2:], label=r"$\Omega_\mathrm{CDM}h^2=%g$" % o)
    H = theory.get_Hubble(redshifts)
    ax_h.plot(redshifts, H / (1 + redshifts), label=r"$\Omega_\mathrm{CDM}h^2=%g$" % o)
ax_cl.set_ylabel(r"$\ell(\ell+1)/(2\pi)\,C_\ell\;(\mu \mathrm{K}^2)$")
ax_cl.set_xlabel(r"$\ell$")
ax_cl.legend()
ax_h.set_ylabel(r"$H/(1+z)\;(\mathrm{km}/\mathrm{s}/\mathrm{Mpc}^{-1})$")
ax_h.set_xlabel(r"$z$")
ax_h.legend()
plt.savefig("omegacdm.png")
# plt.show()
