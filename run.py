import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy.stats import permutation_test, false_discovery_control, spearmanr
import pybedtools

from warnings import filterwarnings
filterwarnings("ignore", category=pd.errors.DtypeWarning)
filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

#set up paths
#=========================
workdir = '/home/AD/rkgadde/L1IP'
annotation = f'{workdir}/hg38_data/annotations/gencode.v46.basic.annotation.sorted.genes.gff3'
celltypes = f'{workdir}/celltypes.txt'

genedir = f'{workdir}/gene_data'
figdir = f'{workdir}/results/CZI/plots'




me_type = 'L1'
abs_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_abs.tsv'
ins_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_ins.tsv'

abs_df = pd.read_csv(abs_file, sep='\t', usecols=[0,1,2,3,4,5,6,7,8], names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])
abs_df['me_type'] = 'absence'
ins_df = pd.read_csv(ins_file, sep='\t', usecols=[0,1,2,3,4,5,6, 7,8],  names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])
ins_df['me_type'] = 'insertion'
l1_df = pd.concat([abs_df, ins_df])
l1_df['length'] = l1_df['end'] - l1_df['start']
emp_ = np.empty(len(l1_df)).astype(str)
emp_.fill('both')
emp_[l1_df['het'].isna()] = 'hom'
emp_[l1_df['hom'].isna()] = 'het'
l1_df['genotype'] = emp_


me_type = 'Alu'
abs_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_abs.tsv'
ins_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_ins.tsv'

abs_df = pd.read_csv(abs_file, sep='\t', usecols=[0,1,2,3,4,5,6,7,8], names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])
abs_df['me_type'] = 'absence'
ins_df = pd.read_csv(ins_file, sep='\t', usecols=[0,1,2,3,4,5,6, 7,8],  names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])
ins_df['me_type'] = 'insertion'
alu_df = pd.concat([abs_df, ins_df])
alu_df['length'] = alu_df['end'] - alu_df['start']
emp_ = np.empty(len(alu_df)).astype(str)
emp_.fill('both')
emp_[alu_df['het'].isna()] = 'hom'
emp_[alu_df['hom'].isna()] = 'het'
emp_[(alu_df['hom'].isna()) & (alu_df['het'].isna())] = 'both'
alu_df['genotype'] = emp_


alu_bt = pybedtools.BedTool.from_dataframe(alu_df)
l1_bt = pybedtools.BedTool.from_dataframe(l1_df)

cgi_df = pd.read_csv('/cndd3/dburrows/DATA/annotations/gencode/gencode.v37.CGI.hg38.jofan.bed', sep='\t',header=None)
all_df = pd.read_csv('/cndd3/dburrows/DATA/annotations/gencode/red.bed', sep='\t',header=None)
# intron_df=all_df[all_df[3].str.contains('intron')]
# nonintron_df=all_df[~all_df[3].str.contains('intron')]
cgi_bt = pybedtools.BedTool.from_dataframe(cgi_df)
all_bt = pybedtools.BedTool.from_dataframe(all_df)


# do shuffle test
reg_names = np.append(all_df[3].unique(), cgi_df[3].unique())
#randomly shuffle
value_l, class_l, reg_l = [],[],[]
tot_df = pd.DataFrame()
for it in range(1000):
    print(it)
    #redo with actual chromosome positions
    alushuff_df = pd.DataFrame()
    for ch in alu_df['chrom'].unique():
        curr = alu_df[alu_df['chrom'] == ch].copy()
        mini = 0
        maxi = np.max(curr['end'])
        length = curr['end'].values - curr['start'].values
        new_start = np.random.randint(mini, maxi - length) 
        new_end = new_start+length
        curr['start'] = new_start
        curr['end'] = new_end
        alushuff_df = pd.concat([alushuff_df, curr])
    
    l1shuff_df = pd.DataFrame()
    for ch in l1_df['chrom'].unique():
        curr = l1_df[l1_df['chrom'] == ch].copy()
        mini = 0
        maxi = np.max(curr['end'])
        length = curr['end'].values - curr['start'].values
        new_start = np.random.randint(mini, maxi - length)
        new_end = new_start+length
        curr['start'] = new_start
        curr['end'] = new_end
        l1shuff_df = pd.concat([l1shuff_df, curr])
    
    alushuff_bt = pybedtools.BedTool.from_dataframe(alushuff_df)
    l1shuff_bt = pybedtools.BedTool.from_dataframe(l1shuff_df)
    
    l1shuff_cgi = l1shuff_bt.intersect(cgi_bt, f=0.3, wo=True) 
    alushuff_cgi = alushuff_bt.intersect(cgi_bt, f=0.3, wo=True) 
    alushuff_all = alushuff_bt.intersect(all_bt, f=0.3, wo=True) 
    l1shuff_all = all_bt.intersect(l1shuff_bt, f=0.3, wo=True) 

    l1shuff_cgi_df = l1shuff_cgi.to_dataframe(disable_auto_names=True, header=None)
    alushuff_cgi_df = alushuff_cgi.to_dataframe(disable_auto_names=True, header=None)
    alushuff_all_df = alushuff_all.to_dataframe(disable_auto_names=True, header=None)
    l1shuff_all_df = l1shuff_all.to_dataframe(disable_auto_names=True, header=None)

    if len(l1shuff_cgi_df) > 0: l1_out = pd.concat([l1shuff_cgi_df.groupby(13).count()[0],l1shuff_all_df.groupby(3).count()[0]])
    else: l1_out = pd.DataFrame(l1shuff_all_df.groupby(3).count()[0])
    alu_out = pd.concat([alushuff_cgi_df.groupby(14).count()[0],alushuff_all_df.groupby(14).count()[0]])
    l1_out = l1_out.reindex(reg_names, fill_value=0).reset_index()
    alu_out = alu_out.reindex(reg_names, fill_value=0).reset_index()
    l1_out['class'] = 'l1'
    alu_out['class'] = 'alu'
    l1_out.rename(columns={'index':'region', 0:'count'}, inplace=True)
    alu_out.rename(columns={14:'region', 0:'count'}, inplace=True)
    comb_df = pd.concat([l1_out, alu_out])
    tot_df = pd.concat([tot_df, comb_df])