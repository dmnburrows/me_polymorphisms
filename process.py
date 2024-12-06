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

import statsmodels.api as sm
import pandas as pd

#===============================================================================
def run_regression(group):
#===============================================================================
    """
    Perform linear regression for a single group in a grouped DataFrame.

    This function fits an Ordinary Least Squares (OLS) regression model for a specific group of data,
    using 'age_encoded', 'sex_encoded', and 'me_type' as predictor variables to predict 'cpm'.
    It returns the regression coefficients and their corresponding p-values.

    Parameters:
    ----------
    group : pandas.DataFrame
        A DataFrame containing the subset of data for one group, with the following required columns:
        - 'age_encoded': Encoded age variable (e.g., 0 for "young", 1 for "adult").
        - 'sex_encoded': Encoded sex variable (e.g., 0 for "female", 1 for "male").
        - 'me_type': Continuous or categorical predictor variable.
        - 'cpm': Continuous response variable (e.g., gene expression level).

    Returns:
    -------
    results : pandas.DataFrame
        A DataFrame containing:
        - 'coefficients': Estimated coefficients for each predictor and the intercept.
        - 'p_values': P-values for each coefficient, testing the null hypothesis that the coefficient is zero.
    
    Notes:
    -----
    - Rows with missing values are dropped before fitting the model.
    - If the group is empty after dropping NaN rows, the function returns `None`.
    - The function assumes the predictor columns are already encoded as numeric values.

    Example:
    -------
    # Example grouped DataFrame
    data = {
        'celltype': ['A', 'A', 'B', 'B'],
        'age_encoded': [1, 2, 1, 2],
        'sex_encoded': [0, 1, 0, 1],
        'me_type': [0.5, 0.8, 0.7, 1.0],
        'cpm': [2.0, 2.8, 3.2, 3.9]
    }
    df = pd.DataFrame(data)

    # Group by 'celltype' and apply the regression
    results = df.groupby('celltype').apply(run_regression)

    # Display results
    print(results)
    """
    # Drop NaN rows to handle missing data
    group = group.dropna()
    if group.empty:
        return None  # Skip empty groups
    
    # Define predictors and response
    X = group[['age_encoded', 'sex_encoded', 'me_type']]
    X = sm.add_constant(X)  # Add intercept
    y = group['cpm']
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Combine coefficients and p-values into a DataFrame
    results = pd.DataFrame({
        'coefficients': model.params,
        'p_values': model.pvalues
    })
    
    return results  # Return the combined DataFrame

#===============================================================================
def fdr(group):
#===============================================================================

    """
    Perform FDR correction (Benjamini-Hochberg) on p-values in a grouped DataFrame.

    Parameters:
    ----------
    group : pandas.DataFrame
        A DataFrame containing at least the following columns:
        - 'p_values': Raw p-values to correct.
        - 'coefficients': Coefficients associated with the p-values.
        - 'contrast': Contrast labels for the coefficients.

    Returns:
    -------
    results : pandas.DataFrame
        A DataFrame containing:
        - 'coefficients': Original coefficients.
        - 'p_values': Original p-values.
        - 'contrast': Contrast labels.
        - 'padj': FDR-adjusted p-values.
    """
    from statsmodels.stats.multitest import multipletests
    nanbool = np.isnan(group['p_values'])
    group = group[~nanbool]
    pv = group['p_values']
    reject, pvals_corrected, _, _ = multipletests(pv, alpha=0.05, method='fdr_bh')
    results = pd.DataFrame({
        'coefficients': group['coefficients'],
        'p_values': group['p_values'],
        'contrast' : group['contrast'],
        'padj': pvals_corrected
    })
    return results


