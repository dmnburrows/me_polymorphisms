n_iter = 1000

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

#process l1
me_type = 'L1'
abs_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_abs.tsv'
ins_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_ins.tsv'
abs_df = pd.read_csv(abs_file, sep='\t', usecols=[0,1,2,3,4,5,6,7,8], names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])[1:]
abs_df['me_type'] = 'absence'
ins_df = pd.read_csv(ins_file, sep='\t', usecols=[0,1,2,3,4,5,6, 7,8],  names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])[1:]
ins_df['me_type'] = 'insertion'
l1_df = pd.concat([abs_df, ins_df])
l1_df['class'] = 'l1'

#process alu
me_type = 'Alu'
abs_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_abs.tsv'
ins_file = f'{workdir}/mC_data/CZI/type/vars/all_{me_type}_ins.tsv'
abs_df = pd.read_csv(abs_file, sep='\t', usecols=[0,1,2,3,4,5,6,7,8], names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])[1:]
abs_df['me_type'] = 'absence'
ins_df = pd.read_csv(ins_file, sep='\t', usecols=[0,1,2,3,4,5,6, 7,8],  names=['chrom','start','end','id', 'length', 'strand', 'class', 'het', 'hom'])[1:]
ins_df['me_type'] = 'insertion'
alu_df = pd.concat([abs_df, ins_df])
alu_df['class'] = 'alu'

#combine + add labels
#===============================
comb_df = pd.concat([l1_df, alu_df])
comb_df['start'] = comb_df['start'].astype(int)
comb_df['end'] = comb_df['end'].astype(int)
comb_df['length'] = comb_df['end'] - comb_df['start']
comb_df['het'] = comb_df['het'].fillna('NaN').astype(str)
comb_df['hom'] = comb_df['hom'].fillna('NaN').astype(str)
comb_df.reset_index(drop=True, inplace=True)
het_bool = np.asarray([True if 'NaN' not in i else False for i in comb_df['het']])
hom_bool = np.asarray([True if 'NaN' not in i else False for i in comb_df['hom']])
het_count = np.asarray([len(i.split(',')) for i in comb_df['het']])
hom_count = np.asarray([len(i.split(',')) for i in comb_df['hom']])
ratio = het_count/hom_count
#set to NaN and inf
ratio[(het_bool == False) & (hom_bool==False)] = np.nan
ratio[(het_bool == True) & (hom_bool==False)] = np.inf
ratio[(het_bool == False) & (hom_bool==True)] = -1*np.inf
comb_df['het_over_hom'] = ratio
# label as majority het or hom, if >2x
genotype = np.empty(len(comb_df)).astype(str)
genotype[ratio >= 2] = 'het'
genotype[ratio <= 0.5] = 'hom'
genotype[(ratio <= 2) & (ratio >= 0.5)] = 'mixed'
genotype[np.isnan(ratio)] = 'NaN'
comb_df['genotype'] = genotype
#label as truncated or full length
trunc = np.empty(len(comb_df), dtype=object)
l1_mask = comb_df['class'] == 'l1'
trunc[l1_mask & (comb_df['length'] > 5500)] = 'full_length'
trunc[l1_mask & (comb_df['length'] <= 5500)] = 'truncated'
alu_mask = comb_df['class'] == 'alu'
trunc[alu_mask & (comb_df['length'] > 280)] = 'full_length'
trunc[alu_mask & (comb_df['length'] <= 280)] = 'truncated'
comb_df['insertion_category'] = trunc


#bedconvert and load annotations
alu_bt = pybedtools.BedTool.from_dataframe(alu_df)
l1_bt = pybedtools.BedTool.from_dataframe(l1_df)
cgi_df = pd.read_csv('/cndd3/dburrows/DATA/annotations/gencode/gencode.v37.CGI.hg38.jofan.bed', sep='\t',header=None)
all_df = pd.read_csv('/cndd3/dburrows/DATA/annotations/gencode/red.bed', sep='\t',header=None)
cgi_bt = pybedtools.BedTool.from_dataframe(cgi_df)
all_bt = pybedtools.BedTool.from_dataframe(all_df)

l1_df = comb_df[comb_df['class']=='l1'].copy()
alu_df = comb_df[comb_df['class']=='alu'].copy()

# do shuffle test
reg_names = np.append(all_df[3].unique(), cgi_df[3].unique())
#randomly shuffle
value_l, class_l, reg_l = [],[],[]
tot_df = pd.DataFrame()
for it in range(n_iter):
    print(it)
    #redo with actual chromosome positions
    alushuff_df = pd.DataFrame()
    for ch in alu_df['chrom'].unique():
        curr = alu_df[alu_df['chrom'] == ch].copy()
        curr['start'] = curr['start'].astype(int)
        curr['end'] = curr['end'].astype(int)
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
        curr['start'] = curr['start'].astype(int)
        curr['end'] = curr['end'].astype(int)
        mini = 0
        maxi = np.max(curr['end'])
        length = curr['end'].values - curr['start'].values
        new_start = np.random.randint(mini, maxi - length)
        new_end = new_start+length
        curr['start'] = new_start
        curr['end'] = new_end
        l1shuff_df = pd.concat([l1shuff_df, curr])

    #do intersection
    #=======================
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
 
    # if len(l1shuff_cgi_df) > 0: l1_out = pd.concat([l1shuff_cgi_df.groupby(13).count()[0],l1shuff_all_df.groupby(3).count()[0]])
    # else: l1_out = pd.DataFrame(l1shuff_all_df.groupby(3).count()[0])
    import itertools
    
    #l1 organise + fill emptys
    cgi = pd.DataFrame(l1shuff_cgi_df.groupby([16, 9, 11,12]).count()[0])
    cgi.reset_index(inplace=True)
    cgi.rename(columns={16:'region', 9: 'me_type', 11: 'genotype', 12:'length', 0:'count'}, inplace=True)
    rest = pd.DataFrame(l1shuff_all_df.groupby([3, 15, 17,18]).count()[0])
    rest.reset_index(inplace=True)
    rest.rename(columns={3:'region', 15: 'me_type', 17: 'genotype', 18:'length', 0:'count'}, inplace=True)
    l1_prac = pd.concat([rest, cgi])
    regions = reg_names
    me_types = l1_prac['me_type'].unique()
    genotypes = l1_prac['genotype'].unique()
    lengths = l1_prac['length'].unique()
    
    # Generate all possible combinations
    all_combinations = list(itertools.product(regions, me_types, genotypes, lengths))
    full_df = pd.DataFrame(all_combinations, columns=['region', 'me_type', 'genotype', 'length'])
    l1merged_df = full_df.merge(l1_prac, on=['region', 'me_type', 'genotype', 'length'], how='left')
    l1merged_df['count'] = l1merged_df['count'].fillna(0).astype(int)
    
    #alu organise + fill emptys
    cgi = pd.DataFrame(alushuff_cgi_df.groupby([16, 9, 11,12]).count()[0])
    cgi.reset_index(inplace=True)
    cgi.rename(columns={16:'region', 9: 'me_type', 11: 'genotype', 12:'length', 0:'count'}, inplace=True)
    rest = pd.DataFrame(alushuff_all_df.groupby([16, 9, 11,12]).count()[0])
    rest.reset_index(inplace=True)
    rest.rename(columns={16:'region', 9: 'me_type', 11: 'genotype', 12:'length', 0:'count'}, inplace=True)
    alu_prac = pd.concat([rest, cgi])
    regions = reg_names
    me_types = alu_prac['me_type'].unique()
    genotypes = alu_prac['genotype'].unique()
    lengths = alu_prac['length'].unique()
    
    # Generate all possible combinations
    all_combinations = list(itertools.product(regions, me_types, genotypes, lengths))
    full_df = pd.DataFrame(all_combinations, columns=['region', 'me_type', 'genotype', 'length'])
    alumerged_df = full_df.merge(alu_prac, on=['region', 'me_type', 'genotype', 'length'], how='left')
    alumerged_df['count'] = alumerged_df['count'].fillna(0).astype(int)
    
    l1merged_df['class'] = 'l1'
    l1merged_df['iteration'] = it
    alumerged_df['class'] = 'alu'
    alumerged_df['iteration'] = it
    tot_df = pd.concat([tot_df, pd.concat([l1merged_df, alumerged_df])])

tot_df.to_csv('/cndd3/dburrows/DATA/me_polymorphisms/analysis/shuffle_enrichment.csv', sep='\t')