from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from astropy.table import Table


# Define constants and paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / '../lc_gap_csv'
OUT_DIR = BASE_DIR / 'ZTF-SN-data-csv'
CLASSIFICATION_DIR = BASE_DIR / '../classified_unclassified_objet/classified_lc_gap_all.csv'
EVENT_FILE = BASE_DIR / '../classified_unclassified_objet/ZTF_Gap_Trasient_classifications.dat'

#Output directory exists
OUT_DIR.mkdir(exist_ok=True)

def find_csv_files(datapath):
    """Return a list of all CSV files in the specified directory."""
    return list(datapath.rglob('*.csv'))

def read_classifications(filepath):
    """Read classifications from a CSV file."""
    classified_data = pd.read_csv(filepath)
    source_names = classified_data['name'].unique()
    return classified_data, source_names
def main():
    classified_data, source_names = read_classifications(CLASSIFICATION_DIR)
    sn_events = Table.read(EVENT_FILE, format='ascii.fast_tab')

    ztf_files = find_csv_files(DATA_PATH)
    # Initialize counters separately for clarity and to avoid dynamic key issues.
    counts = {
        'total_sn_ia': 0,
        'total_sn_ca_rich': 0,
        'total_sn_ii': 0,
        'total_sn_ib': 0,
        'total_sn_ic': 0
    }

    with tqdm(total=len(ztf_files)) as progress:
        for file in ztf_files:
            name = file.stem.split('_')[-1]
            if name in source_names:
                event_index = np.where(sn_events['source_name'] == name)[0][0]
                prob = sn_events[event_index]['probability']
                class_as = sn_events[event_index]['classification']

                data = pd.read_csv(file)
                data.loc[:, 'filter'] = data['filter'].replace({'r': 'ztfr', 'g': 'ztfg', 'i': 'ztfi'})
                df = data[['jd', 'magpsf', 'sigmapsf', 'filter', 'magzpsci', 'programid']].copy()
                df.sort_values("jd", ascending=True, inplace=True)

                if prob > 0.5:
                    processed = False
                    # Folders and corresponding counter keys
                    folders = {
                        'Ia': ('IaSN', 'total_sn_ia'),
                        'Ca-rich': ('Ca_rich_SN', 'total_sn_ca_rich'),
                        'II': ('IISN', 'total_sn_ii'),
                        'Ib': ('IbIcSN', 'total_sn_ib'),
                        'Ic': ('IbIcSN', 'total_sn_ic')
                    }
                    for key, (folder_name, counter_key) in folders.items():
                        if key in class_as:
                            counts[counter_key] += 1
                            directory = OUT_DIR / folder_name
                            directory.mkdir(exist_ok=True)
                            df.to_csv(directory / f'lc_{name}.csv', index=False)
                            processed = True
                            break
                    if not processed:
                        print(f"Unprocessed or unmatched type for {name} with classification {class_as}.")
                else:
                    print(f"Low probability for {name}: {prob}")

                progress.update()

    print(f"Processed Counts: SN Ia={counts['total_sn_ia']}, Ca-rich={counts['total_sn_ca_rich']}, SN II={counts['total_sn_ii']}, SN Ib={counts['total_sn_ib']}, SN Ic={counts['total_sn_ic']}")

if __name__ == "__main__":
    main()

