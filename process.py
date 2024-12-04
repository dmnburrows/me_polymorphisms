import glob
import os
import pandas as pd
import numpy as np

#===============================================================================
def combine_genes_bycell_filter(rna=None, pattern=None, mini=None, n_samp=None):
#===============================================================================
    """
    Combines gene expression data across multiple files, filtering genes based on a threshold
    of expression level and number of samples, and annotating with cell type information.

    Parameters:
    ----------
    rna : list of str
        A list of file paths to RNA-seq expression data files. Each file should be a tab-separated
        values (TSV) file with genes as rows and samples as columns.
    pattern : str
        A substring pattern in the file names that will be replaced to extract the cell type name.
        For example, if the filenames are 'celltype1_expression.tsv', and `pattern` is '_expression.tsv',
        the resulting cell type will be 'celltype1'.
    mini : float
        The minimum expression value for a gene to be considered expressed in a sample, in CPM.
    n_samp : int
        The minimum number of samples in which a gene must meet the `mini` threshold to be retained.

    Returns:
    -------
    matrices : list of pandas.DataFrame
        A list of DataFrames, one for each input file, containing:
        - The filtered gene expression data, with rows corresponding to individual genes
          and columns corresponding to samples.
        - A 'celltype' column, annotating each row with the corresponding cell type extracted
          from the file name.
    ```
    """

    rna = glob.glob(f'{rna}')

    matrices = []
    mini = np.log2(mini+1) #minimum log2cpm+1 value

    for rna_file in rna:
        celltype = os.path.basename(rna_file).replace(pattern, '')
        m = pd.read_csv(rna_file, sep='\t', index_col=0)
        m = m[np.sum(m > mini, axis=1) > n_samp]
        m = pd.DataFrame(m.T.stack())
        m['celltype'] = celltype
        matrices.append(m)

    cpm = pd.concat(matrices)

    cpm = cpm.reset_index()
    cpm = cpm.rename(columns={'level_0': 'sample', 'level_1': 'gene_id', 0: 'cpm'})
    cpm['gene_id'] = cpm['gene_id'].str.split('.').str[0]

    return(cpm)