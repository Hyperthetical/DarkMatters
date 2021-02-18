import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = np.loadtxt("/home/geoff/Dropbox/Physics/Wimps/out/reticulum2_bb_mx100GeV_z6.8e-06_b1.0_exp_d1_3.1e+28_burkert_none_radio_rflux_sb_1.4e+03MHz.out",unpack=True)

convertAngles = 3437.75
microJy = 1e6
sigv = 2.2e-26
plt.yscale("log")
plt.xscale("log")
plt.ylim([1e-1,1e3])
plt.xlim([0.1,5.0])
plt.ylabel(r"I($\Theta$) ($\mu$Jy beam$^{-1}$)",fontsize=16)
plt.xlabel(r"$\Theta$ (arcmin)",fontsize=16)
plt.plot(data[0]*convertAngles,data[1]*microJy*sigv/0.03)
plt.savefig("surface_brightness_retII.pdf",format="pdf")
plt.tight_layout()
plt.show()

data = np.loadtxt("/home/geoff/Dropbox/Physics/Wimps/out/reticulum2_bb_mx100GeV_z6.8e-06_b1.0_exp_d1_3.1e+28_ein0.18_none_radio_radial_1.4e+03MHz.out",unpack=True)
uplims = np.loadtxt("/home/geoff/Dropbox/Data/regis2017_j.data",unpack=True)

uplimIntp = interp1d(uplims[0],uplims[1])
newX = np.logspace(np.log10(uplims[0][0]),np.log10(uplims[0][-1]*0.999999),num=50)
uplims = [newX,uplimIntp(newX)]

microJy = 1e6
sigv = 3e-26
angArea = 0.03/(np.arctan(data[0]*1e3/30.0)**2*convertAngles**2)
print(angArea)
plt.yscale("log")
#plt.xscale("log")
plt.ylim([1e0,1e3])
plt.xlim([0.1,5.0])
plt.ylabel(r"$S(\nu,\Theta \leq \Theta_s)$ ($\mu$Jy beam$^{-1}$)",fontsize=16)
plt.xlabel(r"$\Theta_s$ (arcmin)",fontsize=16)
plt.plot(np.arctan(data[0]*1e3/30.0)*convertAngles,data[1]*microJy*sigv*angArea)
text = r"$m_\chi = 100$ GeV"+"\n"+r"$\chi\chi \to b\bar{b}$"+"\n"+r"$\langle\sigma V\rangle = 3\times 10^{26}$ cm$^3$ s$^{-1}$"
plt.text(3.0,2e1,text)
#plt.errorbar(uplims[0],uplims[1]*microJy,yerr=uplims[1]*0.3*microJy,uplims=True,marker="o",linestyle="none")
plt.savefig("radial_flux_retII.pdf",format="pdf")
plt.tight_layout()
plt.show()