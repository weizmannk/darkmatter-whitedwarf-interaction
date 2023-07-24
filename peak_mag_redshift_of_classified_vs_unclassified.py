import os
import numpy as np
import pandas as pd
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

datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/classified_unclassified_objet"

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


for gap in ['classified', 'unclassified']:
    if gap == 'classified':
        classified_data   = pd.read_csv(f'{datapath}/classified_lc_gap_all.csv')
        classified_data.groupby('name').ngroups

        #hist, bins = np.histogram(classified_data.groupby(['name']).apply(lambda x: x.shape[0]),bins=np.arange(-0.5,10.,1))
        
        ##Photometry peak mag distribustion
        hist, bins = np.histogram(classified_data.groupby(['name']).apply(lambda x: x['magpsf'].min()),bins=np.arange(13,24,0.5))
        sax[0].step(bins, np.pad(hist, (1, 0)) / hist.max(),color=class_color, lw=1, label='classified')
        
        clu_class = pd.read_csv(f'{datapath}/crossmatch_clu_classified.csv')
        clu_class.groupby('name').ngroups
        ##CLU redshift classified distribustion  
        hist, bins = np.histogram(clu_class.groupby(['name']).apply(lambda x: x['clu_z'].median()),bins=np.arange(-1e-6,0.06,0.001))
        sax[1].step(bins, np.pad(hist, (1, 0)) / hist.max(),color=class_color, lw=1, label='classified')
        

    else:
        
        unclassified_data = pd.read_csv(f'{datapath}/unclassified_lc_gap_all.csv')
        unclassified_data.groupby('name').ngroups

        #hist, bins = np.histogram(unclassified_data.groupby(['name']).apply(lambda x: x.shape[0]),bins=np.arange(-0.5,10.,1))
        hist, bins = np.histogram(unclassified_data.groupby(['name']).apply(lambda x: x['magpsf'].min()),bins=np.arange(13,24,0.5))
        sax[0].step(bins, np.pad(hist, (1, 0)) / hist.max(),color=unclass_color, lw=1, label='unclassified')
        
                
        clu_unclass = pd.read_csv(f'{datapath}/crossmatch_clu_unclassified.csv')
        clu_unclass.groupby('name').ngroups
        ##CLU redshift unclassified distribustion 
        hist, bins = np.histogram(clu_unclass.groupby(['name']).apply(lambda x: x['clu_z'].median()),bins=np.arange(-1e-6,0.06,0.001))
        sax[1].step(bins, np.pad(hist, (1, 0)) / hist.max(),color=unclass_color, lw=1, label='unclassified')




plt.suptitle('Peak mag and Redshift of Classified vs Unclassified', size=26,  fontname="Times New Roman")
    
sax[0].legend(loc='upper left')
sax[0].set_xlabel('peak mag')
sax[0].get_yaxis().set_visible(False)

sax[1].legend(loc='best')
sax[1].set_xlabel('readshift distribustion')
sax[1].get_yaxis().set_visible(False)



plt.gcf().set_size_inches(12, 9)
plt.subplots_adjust(left=0.1,
        bottom=0.6,
        right=0.9,
        top=0.9,
        wspace=0.4,
        hspace=0.4)
plt.tight_layout()
plt.savefig(f'Redshift_Peak_magnitude.png')
plt.close()