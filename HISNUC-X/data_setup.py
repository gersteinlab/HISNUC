
import os
import pandas as pd
from logger_config import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
from io import BytesIO
import gzip
from scipy import stats
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    csv_file_path: str,
    image_dir_path: str,
    qupath_feature_path: str,
    metadata_file: str,
    tissue_name: str,
    tau_threshold: float,
    expression_threshold: float,
    num_genes: int,
    seed: int,
):
    logger.info("\n")
    logger.info("Preprocessing data...\n")
    #### loading gene expression data
    gene_expression_data = pd.read_csv(csv_file_path).set_index("index")
    logger.info(f"Gene expression csv file data shape: {gene_expression_data.shape}")

    #### loading image filenames
    image_files = [i.split("_")[0] for i in os.listdir(image_dir_path)]
    image_filenames_df = pd.DataFrame(image_files, columns=['image_file']).set_index('image_file')
    logger.info(f"Image files data shape: {image_filenames_df.shape}")

    #### merge the gene expression data with image filenames
    merged_data = pd.concat([image_filenames_df, gene_expression_data], axis=1, join="inner")
    merged_data.columns = merged_data.columns.str.split(".").str[0]
    logger.info(f"Gene expression data merged with image filenames shape: {merged_data.shape}")


    #### download and load tissue-specific gene expression data
    url = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    response = requests.get(url)
    expression_data = gzip.decompress(response.content)
    expression_df = pd.read_csv(BytesIO(expression_data), sep='\t', skiprows=2)

    #### load tissue-specific gene data
    tissue_specific_data = pd.read_csv("/gpfs/gibbs/pi/gerstein/rm2586/GTEx-imaging-prediction/tau-gtexv8.csv")
    tissue_specific_data = tissue_specific_data.merge(expression_df, how="inner", left_on="Unnamed: 0", right_on="Name")
    tissue_specific_data = tissue_specific_data[["Name", "x", tissue_name]]
    tissue_specific_data.Name = tissue_specific_data.Name.str.split(".").str[0]

    #### select the tissue-specific genes
    selected_genes = tissue_specific_data[(tissue_specific_data["x"] > tau_threshold) & (tissue_specific_data[tissue_name] > expression_threshold)]
    logger.info(f"Number of tissue-specific genes over expression_threshold: {len(selected_genes)}")
    selected_genes = selected_genes.sort_values(by=tissue_name, ascending=False).head(num_genes)

    #### merging tissue-specific genes with merged data (gene expression data with image filenames)
    merged_data_with_genes = merged_data[merged_data.columns.intersection(selected_genes['Name'])]
    individual_id_df = pd.DataFrame(index=merged_data_with_genes.index)
    individual_id_df["individual ID"] = merged_data_with_genes.index.str.split('-').str[:2].str.join('-')
    merged_data_with_genes = pd.concat([merged_data_with_genes, individual_id_df], axis=1)
    logger.info(f"Merged data with genes shape: {merged_data_with_genes.shape}")
    
    #### load and merge qupath features with image filenames
    qupath_features = pd.read_csv(qupath_feature_path).set_index("Image")
    nuc_first_n_rows_data = qupath_features.iloc[:,:]
    merged_data_with_features = pd.concat([image_filenames_df, qupath_features], axis=1, join="inner")
    merged_data_with_features['individual ID'] = merged_data_with_features.index.str.split('-').str[:2].str.join('-')
    merged_data_with_features['image_file'] = merged_data_with_features.index
    logger.info(f"Merged data with features shape: {merged_data_with_features.shape}")
   
    sample_meta=pd.read_csv("/gpfs/gibbs/pi/gerstein/rm2586/GTEx-genotype/GTEx_Analysis_2017-06-05_v8_Annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt", delimiter="\t")
    sample_meta=sample_meta[["SAMPID","SMRIN","SMATSSCR"]]
    sample_meta["SAMPID"]=sample_meta["SAMPID"].str.split('-').str[:3].str.join('-')
    sample_meta=sample_meta.merge(image_filenames_df, how="inner", left_on="SAMPID", right_on=image_filenames_df.index)
    sample_meta["SAMPID"]=sample_meta["SAMPID"].str.split('-').str[:2].str.join('-')
    
    #### load and preprocess metadata
    metadata = pd.read_csv(metadata_file).set_index("SUBJID")


    categorical_vars = ['SEX', 'RACE', 'DTHHRDY']
    for var in categorical_vars:
        if var == 'SEX':
            prefix = 'Sex'
        else:
            prefix = var
        one_hot_encoded = pd.get_dummies(metadata[var], prefix=prefix)
        one_hot_encoded = one_hot_encoded.astype(int)
        metadata = pd.concat([metadata.drop(var, axis=1), one_hot_encoded], axis=1)
    print("**************metadata")

    
    metadata=metadata.merge(sample_meta, how="inner", right_on="SAMPID", left_on=metadata.index)
    metadata = metadata.dropna()
    print("metadata.index")
    print(metadata.columns)
    
    categorical_vars = ['SMATSSCR']
    for var in categorical_vars:
        if var == 'SEX':
            prefix = 'Sex'
        else:
            prefix = var
        one_hot_encoded = pd.get_dummies(metadata[var], prefix=prefix)
        one_hot_encoded = one_hot_encoded.astype(int)
        metadata = pd.concat([metadata.drop(var, axis=1), one_hot_encoded], axis=1)

    #### merging metadata with merged data
    merged_data_with_metadata = merged_data_with_features.merge(metadata, how="inner", right_on="SAMPID", left_on="individual ID")
    logger.info(f"Number of NaN values in merged data with metadata: {merged_data_with_metadata.isna().sum().sum()}")
    logger.info(f"Merged data with metadata shape: {merged_data_with_metadata.shape}")

    #### relevant columns selection
    nucleus_features=['Nucleus: Area_max500', 'Nucleus: Area std',
       'Nucleus: Area Q1', 'Nucleus: Area Q2',
       'Nucleus: Area Q3', 'Nucleus: Area trimmed_mean90',
       'Nucleus: Circularity_max500',
       'Nucleus: Circularity std',
       'Nucleus: Circularity Q1',
       'Nucleus: Circularity Q2', 'Nucleus: Circularity Q3',
       'Nucleus: Circularity trimmed_mean90','Nucleus: Eccentricity_max500',
       'Nucleus: Eccentricity std',
       'Nucleus: Eccentricity Q1',
       'Nucleus: Eccentricity Q2', 'Nucleus: Eccentricity Q3',
       'Nucleus: Eccentricity trimmed_mean90',
       'Nucleus/Cell area ratio_max500',
       'Nucleus/Cell area ratio std',
       'Nucleus/Cell area ratio Q1', 'Nucleus/Cell area ratio Q2',
       'Nucleus/Cell area ratio Q3', 'Nucleus/Cell area ratio trimmed_mean90']
        #### dropping nan values in 'Nucleus/Cell area ratio IQR_interval', 'Nucleus/Cell area ratio Q1', 'Nucleus/Cell area ratio Q3'
    merged_data_with_metadata=merged_data_with_metadata.dropna()
    
    selected_columns = ['individual ID','HGHT', 'WGHT', 'BMI', 'TRDNISCH', 
                        'drinkindex', 'smokeindex', 'TRVNTSR', 'SMRIN',
                        'Sex_1', 'Sex_2', 
                        'RACE_1','RACE_2', 'RACE_3', 'RACE_4', 'RACE_98', 'RACE_99', 
                        'DTHHRDY_0.0', 'DTHHRDY_1.0', 'DTHHRDY_2.0', 'DTHHRDY_3.0', 'DTHHRDY_4.0','SMATSSCR_0.0', 'SMATSSCR_1.0', 'SMATSSCR_2.0',
                        'AGE','image_file'] + nucleus_features

    merged_data_with_metadata = merged_data_with_metadata[selected_columns]

    #### merging with tissue-specific gene data
    final_merged_data = merged_data_with_metadata.merge(merged_data_with_genes, how="inner", on="individual ID")
    final_merged_data = final_merged_data.drop_duplicates(subset='individual ID', keep='first')
    logger.info(f"Final merged data shape: {final_merged_data.shape}")

    import numpy as np
    #### scaling the continuous variables + nucleus_features
    continuous_vars = ['HGHT', 'WGHT', 'BMI', 'TRDNISCH','drinkindex', 'smokeindex','AGE','SMRIN']
    for var in continuous_vars:
        scaler = StandardScaler()
        final_merged_data[var] = scaler.fit_transform(np.array(final_merged_data[var]).reshape(-1, 1))
    for feature in nucleus_features:
        scaler = StandardScaler()
        final_merged_data[feature] = scaler.fit_transform(np.array(final_merged_data[feature]).reshape(-1, 1))

    shuffled_data = final_merged_data.sample(frac=1, random_state=seed)
    logger.info(f"Final shuffled data:\n{shuffled_data}")

        
    # Step 1: Compute Correlation Matrix
    # Assuming 'nucleus_features' is a DataFrame with 5 nucleus-related features and 'gene_expression' is a DataFrame with 100 gene expression features
    nucleus_features=shuffled_data[['Nucleus: Area_max500', 'Nucleus: Area std',
       'Nucleus: Area Q1', 'Nucleus: Area Q2',
       'Nucleus: Area Q3', 'Nucleus: Area trimmed_mean90',
       'Nucleus: Circularity_max500',
       'Nucleus: Circularity std',
       'Nucleus: Circularity Q1',
       'Nucleus: Circularity Q2', 'Nucleus: Circularity Q3',
       'Nucleus: Circularity trimmed_mean90','Nucleus: Eccentricity_max500',
       'Nucleus: Eccentricity std',
       'Nucleus: Eccentricity Q1',
       'Nucleus: Eccentricity Q2', 'Nucleus: Eccentricity Q3',
       'Nucleus: Eccentricity trimmed_mean90',
       'Nucleus/Cell area ratio_max500',
       'Nucleus/Cell area ratio std',
       'Nucleus/Cell area ratio Q1', 'Nucleus/Cell area ratio Q2',
       'Nucleus/Cell area ratio Q3', 'Nucleus/Cell area ratio trimmed_mean90']]
    gene_expression = shuffled_data.iloc[:, -100:]
    

    corr=[]
    corr_df=pd.DataFrame(np.nan, index=range(24), columns=range(100))
    p_df=pd.DataFrame(np.nan, index=range(24), columns=range(100))
    for i in range(24):
        for j in range(100):
            a=stats.spearmanr(nucleus_features.iloc[:,i], gene_expression.iloc[:,j], axis=0, nan_policy='propagate', alternative='two-sided')  
            a=list(a)
            corr=a[0]
            corr_df.iloc[i,j]=corr
            p=a[1]
            p_df.iloc[i,j]=p

    final_corr001=corr_df[p_df<0.01]
    final_corr001["index"]=list(nucleus_features.columns)

    final_corr005=corr_df[p_df<0.05]
    final_corr005["index"]=list(nucleus_features.columns)



    import numpy as np 
    from pandas import DataFrame
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, shrink=0.4)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")


        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(labelsize=5)
        return im, cbar


    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    fig, ax = plt.subplots(figsize=(15, 8))
    Index= list(nucleus_features)
    Cols = list(gene_expression)
    df_c = corr_df[p_df<0.000001].fillna(0)  #30*100=3000
    dfnp=np.array(df_c)
    color_map = plt.cm.get_cmap('RdBu')
    cmap_r=color_map.reversed()
    im, _ = heatmap(dfnp, Index, Cols, ax=ax,cmap=cmap_r, vmin=-0.5, vmax=0.5, cbarlabel="correlation coeff.")

    plt.tight_layout()
    plt.savefig(tissue_name+'_corr-nucleus-gene-expression.png',dpi=1000,bbox_inches = 'tight')



    # Assuming dfnp is your NumPy array
    threshold = 0.0  # Set your threshold here

    # Convert values to 0 or 1 based on the threshold
    binary_array = np.where(abs(dfnp) > threshold, 1, 0)

    # Calculate row sums
    row_sums = np.sum(binary_array, axis=1)
    row_sum_threshold = 25

    # Boolean indexing to filter column names
    selected_nucleus = list(np.array(nucleus_features.columns)[row_sums > row_sum_threshold])
    print(selected_nucleus)
   



    nucleus_feature_cols = ['HGHT', 'WGHT', 'BMI', 'TRDNISCH', 
                            'drinkindex', 'smokeindex', 'TRVNTSR', 'SMRIN',
                            'Sex_1', 'Sex_2', 
                            'RACE_1','RACE_2', 'RACE_3', 'RACE_4', 'RACE_98', 'RACE_99', 
                            'DTHHRDY_0.0','DTHHRDY_1.0', 'DTHHRDY_2.0', 'DTHHRDY_3.0', 'DTHHRDY_4.0', 
                            'AGE','SMATSSCR_0.0', 'SMATSSCR_1.0', 'SMATSSCR_2.0'] + selected_nucleus
    nucleus_feature = shuffled_data[nucleus_feature_cols]
    features = shuffled_data[['image_file']]
    labels = shuffled_data.iloc[:, -num_genes:]
    logger.info(f"Labels:\n{labels}")

    return features, nucleus_feature, labels

def split_data(features, nucleus_feature, labels, test_size, seed):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - features (DataFrame): Image-level identifiers or metadata features (e.g., 'image_file').
    - nucleus_feature (DataFrame): Numerical features related to nucleus morphology.
    - labels (DataFrame): Gene expression targets for prediction.
    - test_size (float): Proportion of data to use as the test set.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple: (X_train, X_nucleus_train, y_train, X_val, X_nucleus_val, y_val, X_test, X_nucleus_test, y_test)
    """
    
    # Step 1: Split into temporary (train+val) and test sets
    X_temp, X_test, X_nucleus_temp, X_nucleus_test, y_temp, y_test = train_test_split(
        features, nucleus_feature, labels,
        test_size=test_size,
        random_state=seed
    )

    # Step 2: Split temp into train and validation sets (20% of temp = 16% of total if test_size=0.2)
    X_train, X_val, X_nucleus_train, X_nucleus_val, y_train, y_val = train_test_split(
        X_temp, X_nucleus_temp, y_temp,
        test_size=0.2,
        random_state=seed
    )

    # Log dataset sizes
    logger.info(f"\nTraining set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")

    return (
        X_train, X_nucleus_train, y_train,
        X_val, X_nucleus_val, y_val,
        X_test, X_nucleus_test, y_test
    )

def create_datasets(
    csv_file_path: str,
    image_dir_path: str,
    qupath_feature_path: str,
    metadata_file: str,
    tissue_name: str,
    tau_threshold: float,
    expression_threshold: float,
    num_genes: int,
    seed: int,
    test_size: float,
):
    """
    Generates preprocessed training, validation, and test datasets from multi-modal inputs.

    Parameters:
    - csv_file_path (str): Path to gene expression CSV file (rows = samples, cols = genes).
    - image_dir_path (str): Directory containing image file names.
    - qupath_feature_path (str): Path to QuPath nucleus feature CSV file.
    - metadata_file (str): Path to metadata CSV with demographic and technical covariates.
    - tissue_name (str): GTEx tissue name used for filtering tissue-specific genes.
    - tau_threshold (float): Tau score threshold for selecting tissue-specific genes.
    - expression_threshold (float): TPM threshold for gene expression.
    - num_genes (int): Number of tissue-specific genes to retain (top-N).
    - seed (int): Random seed for reproducibility.
    - test_size (float): Proportion of the dataset reserved for testing.

    Returns:
    - Tuple:
        - X_train, X_val, X_test: DataFrames of image-level identifiers.
        - X_nucleus_train, X_nucleus_val, X_nucleus_test: DataFrames of nucleus/demographic features.
        - y_train, y_val, y_test: DataFrames of gene expression labels.
    """
    
    # Log parameters for reproducibility
    logger.info(f"Tissue name: {tissue_name}")
    logger.info(f"Tau threshold: {tau_threshold}")
    logger.info(f"Expression threshold: {expression_threshold}")
    logger.info(f"Number of genes: {num_genes}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Test size: {test_size}")

    # Step 1: Preprocess and integrate all inputs
    features, nucleus_feature, labels = preprocess_data(
        csv_file_path,
        image_dir_path,
        qupath_feature_path,
        metadata_file,
        tissue_name,
        tau_threshold,
        expression_threshold,
        num_genes,
        seed
    )

    # Step 2: Split into train, val, and test sets
    X_train, X_nucleus_train, y_train, \
    X_val, X_nucleus_val, y_val, \
    X_test, X_nucleus_test, y_test = split_data(
        features, nucleus_feature, labels, test_size, seed
    )

    return X_train, X_nucleus_train, y_train, X_val, X_nucleus_val, y_val, X_test, X_nucleus_test, y_test
