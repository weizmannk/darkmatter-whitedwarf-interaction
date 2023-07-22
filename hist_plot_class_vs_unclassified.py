import os
from tqdm.auto import tqdm
from pathlib import Path
from astropy.table import join, Table
import glob
import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as  mpatches
from matplotlib import pyplot as plt
#plt.style.use('seaborn-v0_8-darkgrid')

sns.set_style("whitegrid")
#mpl.use("agg")

fig_width_pt = 700.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 2*0.9 * fig_width * golden_mean  # height in inches
fig_size = [11.5, 10.5] #[fig_width, fig_height]
params = {
    "backend": "pdf",
    "axes.labelsize": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": fig_size,
}
mpl.rcParams.update(params)


datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/lc_gap_csv"
Ia_dir =  f"{os.path.dirname(os.path.realpath('__file__'))}/ZTF_Gap_Trasien_classifications.dat"

targetdir ='../ZTF'
if not os.path.isdir(targetdir):
    os.makedirs(targetdir)
   
   
unclass_color, class_color, Ia_class_color = sns.color_palette(
    'rocket', 3)


def data_filename(datapath):
    """[using recursively to find files]

    :param datapath: the filename directory
    :type datapath: [str]
    :return: all .csv files 
    :rtype: [list]
    """
    
    return glob.glob(datapath+'/**/*.csv', recursive=True)



ztf_files = data_filename(datapath)
classified_data = Table.read(Ia_dir, format='ascii.fast_tab')

Ia_sne  = 0
for i in range(len(classified_data)):
    if 'Ia' in classified_data['classification'][i]:
        Ia_sne+=1

unclassified =  len(ztf_files) - len(classified_data)
classified   = len(classified_data)

obj= ['unclassified', 'classified', 'Ia SNe']
X = np.arange(len(obj))
estimation =  [unclassified, classified, Ia_sne]

###Percentages
unclass_per = unclassified/len(ztf_files)*100
class_per = classified/len(ztf_files)*100
Ia_class_per = Ia_sne/len(ztf_files)*100

# Figure Plot 
plt.clf()
fig = plt.figure()
ax = fig.add_subplot()
figsdir = "%s" % (targetdir)

percent = r'\%'
ax.bar(X, estimation,align='center', color = [unclass_color, class_color, Ia_class_color], edgecolor = ['black' for i in estimation], linewidth = 1)

ax.text(X[0],     estimation[0] +200, f'{np.round(unclass_per, 2)} {percent}' , ha = 'center', color='navy', fontsize = 18, fontweight ='bold')
ax.text(X[1],     estimation[1] +10,  f'{np.round(class_per, 2)}    {percent}' , ha = 'center', color='navy', fontsize = 18, fontweight ='bold')
ax.text(X[2],     estimation[2] +10,  f'{np.round(Ia_class_per, 2)} {percent}', ha = 'center', color='navy', fontsize = 18, fontweight ='bold')


handles = [mpatches.Patch(facecolor=unclass_color, label='unclassified'),
           mpatches.Patch(facecolor=class_color, label='classified'),
           mpatches.Patch(facecolor=Ia_class_color, label='Ia SNe')]

ax.legend(handles=handles, shadow=True, loc='best')
ax.set_yscale('log')

ax.set_ylabel(r'number', size=26,  fontname="Times New Roman")
ax.set_xticks(X, obj, size=26,  fontname="Times New Roman")


## Remobe logscale values and reput the values in bellow.
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_yticks([Ia_sne, classified, unclassified])

plt.title('Classified vs Unclassified', size=26,  fontname="Times New Roman")

plotname = os.path.join(figsdir, '%s.png' % 'distribution')
plt.subplots_adjust(hspace=0.1)
plt.savefig(plotname)
plt.close()
