import os
import numpy as np
import bilby

def nugent_hyper_prior(outdir, label='prior', luminosity_distance=None):
    """
    Create and save a prior file using Bilby's PriorDict.

    Parameters:
    outdir (str): Output directory to save the prior file.
    label (str): Base name for the prior file.
    luminosity_distance (float, optional): Luminosity distance to set as a prior.

    Returns:
    str: Path to the saved prior file.
    """
    # Luminosity distance prior
    priors = bilby.core.prior.PriorDict()
    if luminosity_distance:
        priors['luminosity_distance'] = bilby.core.prior.DeltaFunction(
            peak=luminosity_distance, name='luminosity_distance', latex_label='$D_L$', unit='Mpc')
    else:
        priors['luminosity_distance'] = bilby.core.prior.Uniform(
            name='luminosity_distance', minimum=10, maximum=7000, latex_label='$D_L$', unit='Mpc')

    priors['timeshift'] = bilby.core.prior.Uniform(
        minimum=-2.0, maximum=1.0, name='trigger_time', latex_label='$t_0$', unit='s')
    priors['supernova_mag_boost'] = bilby.core.prior.Uniform(
        minimum=-5.0, maximum=5.0, name='supernova_mag_boost', latex_label='$\\rm{SNmag}$')
    prior_path = priors.to_file(outdir=outdir, label=label)
    
    return os.path.join(outdir, f"{label}.prior")


def Piro2021_prior(outdir, label='prior', luminosity_distance=None):
    """
    Create and save a prior file using Bilby's PriorDict for the Piro 2021 model parameters.
    
    Parameters:
    outdir (str): Output directory to save the prior file.
    label (str): Base name for the prior file.

    Returns:
    str: Path to the saved prior file.
    """
    priors = bilby.core.prior.PriorDict()

    # Luminosity distance prior
    if luminosity_distance:
        priors['luminosity_distance'] = bilby.core.prior.DeltaFunction(
            peak=luminosity_distance, name='luminosity_distance', latex_label='$D_L$', unit='Mpc')
    else:
        priors['luminosity_distance'] = bilby.core.prior.Uniform(
            name='luminosity_distance', minimum=10, maximum=1000, latex_label='$D_L$', unit='Mpc')

    priors['timeshift'] = bilby.core.prior.Uniform(
        minimum=-2.0, maximum=1.0, name='trigger_time', latex_label='$t_0$', unit='s')
    
    priors['log10_Menv'] = bilby.core.prior.Uniform(
        name='log10_Menv', minimum=-5.0, maximum=0.0, latex_label='$\\log_{10}M^{\\rm{env}}$', unit='M_sun')
    
    priors['log10_Renv'] = bilby.core.prior.Uniform(
        name='log10_Renv', minimum=10.0, maximum=15.0, latex_label='$\\log_{10}R^{\\rm{env}}$', unit='R_sun')

    priors['log10_Ee'] = bilby.core.prior.Uniform(
        name='log10_Eext', minimum=47.0, maximum=52.0, latex_label='$\\log_{10}E^{\\rm{e}}$', unit='erg')

    prior_path = priors.to_file(outdir=outdir, label=label)
    return os.path.join(outdir, f"{label}.prior")

def salt2_prior(outdir, label='prior', luminosity_distance=None):
    """
    Create and save a prior file using Bilby's PriorDict for the SALT2 model parameters.
    
    Parameters:
    outdir (str): Output directory to save the prior file.
    label (str): Base name for the prior file.
    luminosity_distance (float, optional): Specific luminosity distance to set as a prior.

    Returns:
    str: Path to the saved prior file.
    """
    priors = bilby.core.prior.PriorDict()

    # Luminosity distance prior
    if luminosity_distance:
        priors['luminosity_distance'] = bilby.core.prior.DeltaFunction(
            peak=luminosity_distance, name='luminosity_distance', latex_label='$D_L$', unit='Mpc')
    else:
        priors['luminosity_distance'] = bilby.core.prior.Uniform(
            name='luminosity_distance', minimum=0, maximum=7000, latex_label='$D_L$', unit='Mpc')

    # Time shift prior
    priors['timeshift'] = bilby.core.prior.Uniform(
        minimum=-2.0, maximum=1.0, name='trigger_time', latex_label='$t_0$', unit='s')

    # SALT2 x0 (scale factor) prior
    priors['x0'] = bilby.core.prior.Uniform(
        minimum=0.0, maximum=10.0, name='x0', latex_label='$x_0$')

    # SALT2 x1 (stretch factor) prior
    priors['x1'] = bilby.core.prior.Uniform(
        minimum=-5.0, maximum=5.0, name='x1', latex_label='$x_1$')

    # SALT2 c (color parameter) prior
    priors['c'] = bilby.core.prior.Uniform(
        minimum=-0.5, maximum=2.0, name='c', latex_label='$c$')

    # Save the prior file and return the path
    prior_path = priors.to_file(outdir=outdir, label=label)
    return os.path.join(outdir, f"{label}.prior")


def Sr2023_prior(outdir, label='prior', luminosity_distance=None):
    """
    Create and save a prior file using Bilby's PriorDict for the Sr2023 model parameters.
    
    Parameters:
    outdir (str): Output directory to save the prior file.
    label (str): Base name for the prior file.
    luminosity_distance (float, optional): Specific luminosity distance to set as a prior.

    Returns:
    str: Path to the saved prior file.
    """
    priors = bilby.core.prior.PriorDict()

    # Luminosity distance prior
    if luminosity_distance:
        priors['luminosity_distance'] = bilby.core.prior.DeltaFunction(
            peak=luminosity_distance, name='luminosity_distance', latex_label='$D_L$', unit='Mpc')
    else:
        priors['luminosity_distance'] = bilby.core.prior.Uniform(
            name='luminosity_distance', minimum=1, maximum=500, latex_label='$D_L$', unit='Mpc')

    # Time shift prior
    priors['timeshift'] = bilby.core.prior.Uniform(
        minimum=-2.0, maximum=1.0, name='trigger_time', latex_label='$t_0$', unit='s')

    # Amplitude parameters for different bands
    priors['a_AG_ztfi'] = bilby.core.prior.Uniform(name='a_AG_ztfi', minimum=250, maximum=450)
    priors['a_AG_ztfr'] = bilby.core.prior.Uniform(name='a_AG_ztfr', minimum=100, maximum=200)
    priors['a_AG_ztfg'] = bilby.core.prior.Uniform(name='a_AG_g', minimum=100, maximum=200)

    # Decay rate
    priors['alpha_AG'] = bilby.core.prior.Uniform(name='alpha', minimum=1.01, maximum=2)

    # Flux normalization factors
    priors['f_nu_ztfi'] = bilby.core.prior.Uniform(name='f_nu_ztfi', minimum=1, maximum=2)
    priors['f_nu_ztfr'] = bilby.core.prior.Uniform(name='f_nu_ztfr', minimum=0, maximum=1)
    priors['f_nu_ztfg'] = bilby.core.prior.Uniform(name='f_nu_ztfg', minimum=0, maximum=1)

    # Supernova magnitude modifications
    priors['supernova_mag_boost'] = bilby.core.prior.Gaussian(mu=-0.14, sigma=0.58, name='supernova_mag_boost')
    priors['supernova_mag_stretch'] = bilby.core.prior.TruncatedNormal(
        mu=0.9, sigma=0.3, minimum=0.0, maximum=10.0, name='supernova_mag_stretch')

    # Save the prior file and return the path
    prior_path = priors.to_file(outdir=outdir, label=label)
    return os.path.join(outdir, f"{label}.prior")

def nugent_hyperTrpi2018_prior(outdir, label='prior', luminosity_distance=None):
    """
    Create and save a prior file using Bilby's PriorDict for the Nugent_Trpi2018 model parameters.
    
    Parameters:
    outdir (str): Output directory to save the prior file.
    label (str): Base name for the prior file.
    luminosity_distance (float, optional): Specific luminosity distance to set as a prior.

    Returns:
    str: Path to the saved prior file.
    """
    priors = bilby.core.prior.PriorDict()

    # Luminosity distance prior
    if luminosity_distance:
        priors['luminosity_distance'] = bilby.core.prior.DeltaFunction(
            peak=luminosity_distance, name='luminosity_distance', latex_label='$D_L$', unit='Mpc')
    else:
        priors['luminosity_distance'] = bilby.core.prior.Uniform(
            name='luminosity_distance', minimum=0, maximum=7000, latex_label='$D_L$', unit='Mpc')

    # Redshift prior
    priors['redshift'] = bilby.core.prior.Uniform(
        name='redshift', minimum=0.0, maximum=1.0)

    # Time shift prior
    priors['timeshift'] = bilby.core.prior.Uniform(
        minimum=-2.0, maximum=1.0, name='trigger_time', latex_label='$t_0$', unit='s')

    # Inclination prior (EM)
    priors['inclination_EM'] = bilby.core.prior.Sine(
        name='inclination_EM', minimum=0, maximum=0.8, latex_label='$\\iota$')

    # Logarithmic initial energy prior
    priors['log10_E0'] = bilby.core.prior.Uniform(
        name='log10_E0', minimum=0, maximum=70, latex_label='$\\log_{10}E_{0}$')

    # Core and wing angle priors
    priors['thetaCore'] = bilby.core.prior.Uniform(
        name='thetaCore', minimum=0.01, maximum=np.pi / 2, latex_label='$\\theta_{c}$')
    priors['thetaWing'] = bilby.core.prior.Uniform(
        name='thetaWing', minimum=0.01, maximum=np.pi / 2, latex_label='$\\theta_{w}$')

    # Logarithmic ambient density prior
    priors['log10_n0'] = bilby.core.prior.Uniform(
        name='log10_n0', minimum=-4.9, maximum=10, latex_label='$\\log_{10}n_{0}$')

    # Power law index prior
    priors['p'] = bilby.core.prior.Uniform(
        name='p', minimum=2, maximum=8, latex_label='$p$')

    # Logarithmic microphysical parameters
    priors['log10_epsilon_e'] = bilby.core.prior.Uniform(
        name='log10_epsilon_e', minimum=-4, maximum=0, latex_label='$\\log_{10}\\epsilon_{e}$')
    priors['log10_epsilon_B'] = bilby.core.prior.Uniform(
        name='log10_epsilon_B', minimum=-5, maximum=0, latex_label='$\\log_{10}\\epsilon_{B}$')

    # Supernova magnitude boost prior
    priors['supernova_mag_boost'] = bilby.core.prior.Uniform(
        name='supernova_mag_boost', minimum=-5.0, maximum=5.0, latex_label='$\\rm{SNmag}$')

    # Save the prior file and return the path
    prior_path = priors.to_file(outdir=outdir, label=label)
    return os.path.join(outdir, f"{label}.prior")
