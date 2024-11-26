import pandas as pd
import numpy as np
from glob import glob
import os

def parse_attributes(annot_df):
    """Parse the attributes column of a gff3 file."""
    out_df = annot_df.copy()

    out_df['attributes'] = out_df['attributes'].str.split(';')
    out_df['gene_type'] = out_df['attributes'].str[2].str.replace('gene_type=', '')
    out_df['gene_name'] = out_df['attributes'].str[3].str.replace('gene_name=', '')
    out_df['gene_id'] = out_df['attributes'].str[1].str.replace('gene_id=', '')
    out_df['gene_id'] = out_df['gene_id'].str.split('.').str[0]
    out_df = out_df.drop(columns='attributes')

    return out_df

def get_expressed_genes(rna_path, min_cpm, min_samples):
    """Find all genes such that the gene has a CPM > min_cpm in at least
    min_samples. Return the expressed gene list (may contain duplicates)."""
    rna = glob(f'{rna_path}')
    genes = []

    for rna_file in rna:
        m = pd.read_csv(rna_file, sep='\t', index_col=0)
        m = m[(m > np.log2(min_cpm + 1)).sum(axis=1) >= min_samples]
        genes.append(pd.Series(m.index))

    genes = pd.concat(genes) # may contain duplicates
    genes = genes.str.split('.').str[0]

    return genes

def get_expressed_gene_cpm(rna_path, pattern, min_cpm, min_samples):
    """Find all genes such that the gene has a CPM > min_cpm in at least
    min_samples, and return their expression levels."""
    rna = glob(f'{rna_path}')
    matrices = []

    for rna_file in rna:
        celltype = os.path.basename(rna_file).replace(pattern, '')
        m = pd.read_csv(rna_file, sep='\t', index_col=0)
        m = m[(m > np.log2(min_cpm + 1)).sum(axis=1) >= min_samples]
        m = pd.DataFrame(m.T.stack())
        m['celltype'] = celltype
        matrices.append(m)

    cpm = pd.concat(matrices)

    cpm = cpm.reset_index()
    cpm = cpm.rename(columns={'level_0': 'sample', 'level_1': 'gene_id', 0: 'cpm'})
    cpm['gene_id'] = cpm['gene_id'].str.split('.').str[0]

    return cpm