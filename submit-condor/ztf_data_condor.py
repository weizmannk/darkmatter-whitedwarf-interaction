import os
import glob
import subprocess
import logging
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.time import Time
from astropy.cosmology import Planck15 as cosmo, z_at_value
import bilby

# Setup basic configuration for logging
logging.basicConfig(filename='nmma-simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for cosmological calculations
H0 = 67.9  # Hubble Constant in km/s/Mpc, from Planck 2015 results
Omega_m = 0.31  # Matter density parameter

# Fundamental physical constants
c = 299792458.0  # Speed of light in m/s
G = 6.6743e-11  # Newton's gravitational constant in N(m^2)/(kg^2)

def luminosity_distance_to_redshift(ld):
    """
    Convert luminosity distance to redshift using the Astropy cosmology module.
    Parameters:
        ld (float): Luminosity distance in Mpc.
    Returns:
        float: Computed redshift as a dimensionless quantity.
    """
    return z_at_value(cosmo.luminosity_distance, ld * u.Mpc)

def redshift_to_luminosity_distance(z):
    """
    Calculate luminosity distance for a given redshift based on a cosmological model.
    The formula used comes from http://arxiv.org/pdf/1111.6396v1.pdf

    Parameters:
    z (float): Redshift, dimensionless.

    Returns:
    float: Luminosity distance in Mpc.
    """
    x0 = (1. - Omega_m) / Omega_m
    xZ = x0 / (1. + z)**3

    Phi0 = (1. + 1.320 * x0 + 0.4415 * x0**2 + 0.02656 * x0**3)
    Phi0 /= (1. + 1.392 * x0 + 0.5121 * x0**2 + 0.03944 * x0**3)
    PhiZ = (1. + 1.320 * xZ + 0.4415 * xZ**2 + 0.02656 * xZ**3)
    PhiZ /= (1. + 1.392 * xZ + 0.5121 * xZ**2 + 0.03944 * xZ**3)

    d_luminosity = (2. * c / H0 * (1.0e-3) * (1. + z) / np.sqrt(Omega_m) * (Phi0 - PhiZ / np.sqrt(1. + z)))
    return d_luminosity

def data_filename(datapath):
    """
    Recursively find all .dat files in the specified directory.
    Parameters:
        datapath (str): Path to the directory where data files are stored.
    Returns:
        list: A list of paths to .dat files.
    """
    return glob.glob(os.path.join(datapath, '**', '*.dat'), recursive=True)

def create_prior(prior_file_path, outdir, label='prior', z=None, fix_z=False):
    """
    Create and save a prior file using Bilby's PriorDict.
    Parameters:
        prior_directory (str): Directory where prior files are located.
        outdir (str): Output directory to save the prior file.
        label (str): Base name for the prior file.
        z (float, optional): Redshift to set as a prior.
        fix_z (bool): Whether to fix the redshift at a given value.
    Returns:
        str: Path to the saved prior file.
    """
    
    logging.info(f"Attempting to create prior with path: {prior_file_path} and outdir: {outdir}")
    
    if not os.path.isfile(prior_file_path):
        logging.error(f"Prior file {prior_file_path} does not exist")
        return None
    
    priors = bilby.gw.prior.PriorDict(prior_file_path)
    if fix_z and z is not None:
        from astropy.coordinates import Distance
        distance = Distance(z=z, unit=u.Mpc)
        priors['luminosity_distance'] = bilby.core.prior.DeltaFunction(
            peak=distance.value, name='luminosity_distance', latex_label='$D_L$', unit='Mpc')
    
    prior_output_path = os.path.join(outdir, f"{label}.prior")
    priors.to_file(outdir=outdir, label=label)
    
    return prior_output_path

# Define the model
model = "nugent-hyper"
output_directory = f"OUTPUT/{model.upper()}"
os.makedirs(output_directory, exist_ok=True)
log_dir = os.path.join(output_directory, "logs")
os.makedirs(log_dir, exist_ok=True)

# Data path setup
datapath = os.path.join(os.path.dirname(__file__), 'ZTF-data')
classified_dir = os.path.join(datapath, 'classified_unclassified_objet', 'ZTF_Gap_Trasient_classifications.dat')
classified_sources = Table.read(classified_dir, format='ascii.fast_tab')
data_files = data_filename(os.path.join(datapath, "dat-files"))

# Model based configurations
model_list = [m.strip() for m in model.split(",")]

if "nugent-hyper" in model_list and len(model_list) == 1:
    prior_name = "sncosmo-generic.prior"
elif "Sr2023.prior" in model_list and "nugent-hyper" in model_list:
    prior_name = "Sr2023.prior"
elif "nugent-hyper" in model_list and "TrPi2018" in model_list:
    prior_name = "sncosmo-generic_TrPi2018.prior"
else:
    prior_name =  f"{'_'.join(sorted(model_list))}.prior"
                      
print("Model:", model)
print("Prior Name:", prior_name)

prior_file_path  = os.path.join(os.path.dirname(__file__), 'nmma/priors', prior_name)

job =0

# Process data files
for data_file in data_files:
    candname = os.path.basename(data_file).split('.')[0].split('_')[-1]
    outdir = os.path.join(output_directory, candname)
    os.makedirs(outdir, exist_ok=True)

    # Fetch redshift and convert to luminosity distance
    indx = np.where(classified_sources['source_name'] == candname)[0][0]
    redshift = classified_sources['redshift'][indx]
    
    if np.isnan(redshift):
        logging.info(f"Redshift is NaN for {candname}")
        continue

    prior_file = create_prior(prior_file_path, outdir, label=candname, z=redshift, fix_z=True)
    logging.info(f"Created prior file for model {model} at {prior_file}")
    
    # Set the trigger time
    in_data = pd.read_csv(data_file)
    trigger_time = np.inf
    for line in np.atleast_1d(in_data):
        if np.isinf(float(str(line[0]).split()[3])):
            continue
            
        elif Time(str(line[0]).split()[0], format='isot').mjd -0.5  < trigger_time:
            trigger_time = Time(str(line[0]).split()[0], format='isot').mjd -0.5
      
    label = f"{'_'.join(sorted(model_list))}_{candname}"
                        
     # Setup job arguments for Condor submission #--trigger-time  {trigger_time}
    arguments = f"--model {model} --outdir {outdir} --data {data_file} --prior {prior_file} --trigger-time  {trigger_time} --label {label} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 1024 --Ebv-max 0 --generation-seed 42 --plot --ylim 22,14 --xlim 0,14 --local-only"
    condor_submit_script = f'''
                            universe = vanilla
                            accounting_group = ligo.dev.o4.cbc.pe.bayestar
                            getenv = true
                            executable = {subprocess.check_output(["which", "lightcurve-analysis"]).decode().strip()}
                            arguments = {arguments}
                            output = {log_dir}/$(Cluster)_$(Process).out
                            error = {log_dir}/$(Cluster)_$(Process).err
                            log = {log_dir}/$(Cluster)_$(Process).log
                            request_memory = 8192 MB
                            request_disk = 2000 MB
                            JobBatchName = "Dark-Matter"
                            environment = "OMP_NUM_THREADS=1"
                            queue
                        '''
    with subprocess.Popen(['condor_submit'], text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        stdout, stderr = proc.communicate(input=condor_submit_script)
        if stdout:
            print("Condor submit output:", stdout)
        if stderr or proc.returncode != 0:
            print("Condor submit error:", stderr)
    job+=1

    
print(f"\n\n{job} jobs have been submit.")

