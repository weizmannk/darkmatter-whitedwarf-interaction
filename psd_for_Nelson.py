from gwpy.frequencyseries import FrequencySeries
import os
import logging

"""
# Create an output for the result
outdir =  "./outdir"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Location of 'coinc.xml' , LIGO XML file 
coinc_file =  './coinc.xml'

## Read The psd data in the  LIGO XML file , from each Detector 
psd_H1 = FrequencySeries.read(coinc_file, instrument='H1')
psd_L1 = FrequencySeries.read(coinc_file, instrument='L1')

## Save psd in '.txt' file 
filename = f"{outdir}/L1_psd.txt"
psd_H1.write(target=f"{outdir}/H1_psd.txt", format="txt")
psd_L1.write(target=f"{outdir}/L1_psd.txt", format="txt")

"""




def extract_psds_from_xml(coinc_file, ifos, outdir="."):

    psd_filenames = dict()
    psd_maximum_frequency = None
    for ifo in ifos:
        try:
            psd = FrequencySeries.read(coinc_file, instrument=ifo)
            if psd_maximum_frequency is None:
                psd_maximum_frequency = psd.frequencies.value[-1]
            else:
                psd_maximum_frequency = min(
                    psd_maximum_frequency,
                    psd.frequencies.value[-1],
                )
            filename = f"{outdir}/{ifo}_psd.txt"
            psd.write(target=filename, format="txt")
            psd_filenames[ifo] = filename
        except ValueError:
            logging.warning(f"PSD for {ifo} not found in {coinc_file}.")
    return psd_filenames, psd_maximum_frequency



# Create an output for the result
outdir =  "./outdir"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Location of 'coinc.xml' , LIGO XML file 
coinc_file =  './coinc.xml'

ifos = ['H1', 'L1']
psd_filenames, psd_maximum_frequency = extract_psds_from_xml(coinc_file = coinc_file , ifos=ifos, outdir=outdir)


print(f"filenames : {psd_filenames}, Frequency max :  {psd_maximum_frequency}") 
