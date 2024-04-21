import os
import glob
import numpy as np
from astropy.table import Table
from astropy.time import Time
from tqdm.auto import tqdm
import warnings
from astropy.utils.exceptions import ErfaWarning

# Suppress specific warnings from Astropy that are not critical
warnings.filterwarnings('ignore', category=ErfaWarning)

def find_csv_files(datapath):
    """
    Recursively find all CSV files in the specified directory.
    
    Parameters:
        datapath (str): The path to the directory where CSV files are searched.
        
    Returns:
        list[str]: A list of paths to the found CSV files.
    """
    return glob.glob(os.path.join(datapath, '**', '*.csv'), recursive=True)

def parse_csv(infile, candname, outdir):
    """
    Parse a CSV file to extract photometry data and write it to a .dat file.
    
    Parameters:
        infile (str): Path to the input CSV file.
        candname (str): Candidate name, used as the filename for the output.
        outdir (str): Directory to save the output .dat file.
    
    Returns:
        list[list[str]]: Processed data.
    """
    try:
        # Load data from CSV
        in_data = np.genfromtxt(infile, dtype=None, delimiter=',', skip_header=1, encoding='utf-8')
    except Exception as e:
        print(f"Failed to read {infile}: {e}")
        return []

    out_data = []
    for line in np.atleast_1d(in_data):
        time = Time(line[0], format='jd').isot
        filter = line[3]
        magnitude = line[1]
        error = line[2]

        # Handle special cases for magnitude
        if magnitude == 99.0:
            magnitude = line[4]
            error = np.inf

        out_data.append([time, filter, str(magnitude), str(error)])

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Write data to file
    out_filepath = os.path.join(outdir, f"{candname}.dat")
    with open(out_filepath, 'w') as out_file:
        for line in out_data:
            out_file.write(" ".join(line) + "\n")

    return out_data

def main():
    
    
    for file_path in ['Ca_rich_SN', 'IaSN', 'IISN', 'IbIcSN']:
        base_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(base_dir, 'ZTF-SN-data-csv', file_path)
        target_dir = os.path.join(base_dir, 'ZTF-SN-data-dat', file_path)

        ztf_files = find_csv_files(data_path)
        processed_files_count = 0

        with tqdm(total=len(ztf_files)) as progress:
            for filepath in ztf_files:
                name = os.path.basename(filepath).split('.')[0]
                parse_csv(infile=filepath, candname=name, outdir=target_dir)
                processed_files_count += 1
                progress.update()

        print(f"Processed {processed_files_count} files.")

if __name__ == "__main__":
    main()
