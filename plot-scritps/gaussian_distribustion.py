import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.use('agg')
import matplotlib.gridspec as gridspec


fig_width_pt = 700.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 2*0.9 * fig_width * golden_mean  # height in inches
fig_size = [11.5, 10.5] #[fig_width, fig_height]
params = {
    "backend": "pdf",
    "axes.labelsize": 26,
    "legend.fontsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": fig_size,
}
mpl.rcParams.update(params)

unclass_color,  Ia_class_color, class_color, = sns.color_palette(
    'rocket', 3)

datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/../data/classified_unclassified_objet"

targetdir ='../ZTF'
if not os.path.isdir(targetdir):
    os.makedirs(targetdir)




#band_colors_ztf={'g':'darkgreen','r':'darkred','i':'orange'}#,'limegreen','darkturquoise'

plt.clf()
# Figure Plot
#fig, axs = plt.subplots(nrows=1, ncols=1)

rows,cols=2,1
gs = gridspec.GridSpec(rows,cols)

sax = []
for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))


with tqdm(total=len(['classified', 'unclassified'])) as progress:
        
    for gap in ['classified', 'unclassified']:
        if gap == 'classified':
            
            clu_class = pd.read_csv(f'{datapath}/crossmatch_clu_classified.csv')
            clu_class = clu_class.sort_values("name", ascending=True)
            
            classified_data   = pd.read_csv(f'{datapath}/classified_lc_gap_all.csv')
            #classified_data.groupby('name').ngroups
            
            idx = []
            for ii, name in  enumerate(classified_data['name']):
                if name in list(clu_class['name'].unique()):
                    idx.append(ii)
                
            classified_data = classified_data.iloc[idx]
            classified_data = classified_data.sort_values("name", ascending=True)
            
            peak_mag  = classified_data.groupby(['name']).apply(lambda x: x['magpsf'].min())
            redshift  = clu_class.groupby(['name']).apply(lambda x: x['clu_z'].median())
            
            xy          = np.vstack([redshift, peak_mag])                                     
            z           = gaussian_kde(xy)(xy)
            index       = z.argsort()
            redshift, peak_mag, z = redshift[index], peak_mag[index], z[index]
                                                    
            class_peak_mag_vs_redshift = sax[0].scatter(redshift, peak_mag, c=z, s=25)

        else:
            
            unclassified_data = pd.read_csv(f'{datapath}/unclassified_lc_gap_all.csv')
            clu_unclass = pd.read_csv(f'{datapath}/crossmatch_clu_unclassified.csv')

            idx = []
            for ii, name in  enumerate(unclassified_data['name']):
                if name in list(clu_unclass['name'].unique()):
                    idx.append(ii)
                
            unclassified_data = unclassified_data.iloc[idx]
                    
            peak_mag  = unclassified_data.groupby(['name']).apply(lambda x: x['magpsf'].min())
            redshift  = clu_unclass.groupby(['name']).apply(lambda x: x['clu_z'].median())

            xy          = np.vstack([redshift, peak_mag])                                     
            z           = gaussian_kde(xy)(xy)
            index       = z.argsort()
            redshift, peak_mag, z = redshift[index], peak_mag[index], z[index]
                                                    
            unclass_peak_mag_vs_redshift = sax[1].scatter(redshift, peak_mag, c=z, s=25)
    
        progress.update()        


    plt.suptitle('Gaussian distribustion, Peak mag and Redshift', size=26,  fontname="Times New Roman")
        
    sax[0].legend(['classified'])
    sax[0].set_xlabel(r'Redshift', fontname="Times New Roman", size=24, fontweight="bold")
    sax[0].set_ylabel(r'Peak mag', fontname="Times New Roman", size=24, fontweight="bold")


    sax[1].legend(['Unclassified'])
    sax[1].set_xlabel(r'Redshift', fontname="Times New Roman", size=24, fontweight="bold")
    sax[1].set_ylabel(r'Peak mag', fontname="Times New Roman", size=24, fontweight="bold")


    cbar1 = plt.colorbar(class_peak_mag_vs_redshift,  ax=sax[0])
    cbar2 = plt.colorbar(unclass_peak_mag_vs_redshift,  ax=sax[1])


    plt.gcf().set_size_inches(14, 12)
    plt.subplots_adjust(left=0.1,
            bottom=0.6,
            right=0.9,
            top=0.9,
            wspace=0.4,
            hspace=0.4)
    plt.tight_layout()
    plt.savefig(f'gaussian_distribustion_Redshift_vs_Peak_mag.png')
    plt.close()
