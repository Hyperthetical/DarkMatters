# Dark Matters
Multi-frequency calculations for dark matter annihilation/decay

| Set | Target variable | Value|
|--- | --- | --- |
|name | halo.name | halo name output label |
|rvir | halo.rvir | Virial radius in Mpc, [*]|
|mvir | halo.mvir | Virial mass in solar masses, [*]|			
|rcore | halo.rcore | density profile scale radius in Mpc, [*]|
|r_stellar_half_light | halo.r_stellar_half_light | stellar half-light radius in Mpc [opt]|
|profile | halo.profile | [nfw],burkert,einasto,isothermal,moore,gnfw |
|alpha | halo.alpha | Einasto halo alpha parameter [0.17]|
|dist | halo.dl | luminosity distance to halo in Mpc|
|z | halo.z | halo redshift [0.0]|
|jfactor | halo.J | J-factor in GeV$^2$ cm$^{-5}$| 
|cvir | halo.cvir | Virial concentration [*]|
|b_average | halo.bav | Average magnetic field in micro Gauss [c]|
|ne_average | halo.neav | Average gas density in cm$^{-3}$ [c]|
|rhos | halo.rhos | characteristic rho normalised to critical density [*]|
|rho0 | halo.rho0 | characteristic rho in msol Mpc$^-3$ [*]|
|t_star_formation | halo.t_sf | star formation time in s [opt]|
|submode | sim.sub_mode | sc2006,[none],prada |
|radio_boost | sim.radio_boost_flag | modification for radio [0],1|
|ucmh | halo.ucmh | UCMH flag (0 or 1) [0] |
|jflag | halo.jflag | Normalise DM density to jfactor [0],1|
|nfw_index | halo.gfnw_gamma | $\gamma$ for gnfw profile [1]|
|--- | --- | --- |
|input_spectra_directory | sim.specdir |  directory with dN/dE input spectra|
|nu_flavour | sim.nu_flavour | [all],mu,tau,e|
|mx_set | sim.mx_set | WIMP masses in GeV separated by spaces [\#]|	
|wimp_mode | halo.mode | [ann],decay|
|particle_model | phys.particle_model | label for output [c]|
|channel | phys.channel | Annihilation channels separated by spaces [bb]|
|branching | phys.branching | Branching ratios for channels above [1.0]|
|--- | --- | --- |
|qb | phys.qb | Magnetic field variable [**] default[0.0] |
|b_model | sim.b_flag | exp,follow_ne,[flat],powerlaw|
|B | phys.b0 | magnetic field normalisation in micro Gauss [0.0]|
|d | phys.lc | coherence length for B field in kpc [0.0]|
|delta | phys.delta | Turbulence index [1.666] |
|--- | --- | --- |
|ne_model | sim.ne_model | exp,king,powerlaw,[flat]|
|ne | phys.ne0 | Gas density normalisation cm$^{-3}$ [0.0]|
|qe | phys.qe | Gas variable  [***] default[0.0]|
|ne_scale | phys.lb | gas scale radius in Mpc|
|--- | --- | --- |
|theta | sim.theta | angular radius for flux integration in arcmin|
|r_integrate | sim.rintegrate | Flux integration radius in Mpc|
|flim | sim.flim | min and max frequencies in MHz [1e1 1e5] |
|diff | phys.diff | diffusion flag, 1 or [0] |
|output_directory | sim.out_dir | output directory [./]|
|--- | --- | --- |
|f_num | sim.num | number of frequency samples [40]|
|r_num | sim.n | radial samples, recommend more than 51 (odd) [100]|
|e_bins |  sim.e_bins | bins for dN/dE spectra recommend 71 (odd)|
|gr_num | sim.ngr | radial samples for diffusion, recommend 201 (odd) [300]|
|electrons_from_c | - | [****]|
|--- | --- | --- |
|omega_m | cosmo.w_m | Matter fraction [0.3089]|
|omega_l | cosmo.w_l | Lambda fraction [0.6911]|
|h | cosmo.h | Hubble constant [0.6774] |
|omega_dm | cosmo.w_dm | [0.2589]|
|omega_nu | cosmo.w_nu | [0.0] |
|ps_index | cosmo.n | Structure power-spectrum index [0.968] |
|curvature | cosmo.universe | flat or curved [flat]|
|sigma_8 | cosmo.sigma_8 | Power-spctrum normalisation [0.8159]|
