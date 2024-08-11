# Imports from external modules
from anndata import AnnData
from MultiConfiguration import MDKM
import dynamo as dyn
import muon as mu
from muon import MuData
from muon import atac as ac
import numpy as np
import os
import pandas as pd
import scanpy as sc
import scrublet
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

# Imports from dynamo
from dynamo.dynamo_logger import (
    LoggerManager,
    main_debug,
    main_exception,
    main_info,
    main_info_insert_adata,
    main_warning,
)
from dynamo.preprocessing.gene_selection import (
    select_genes_monocle
)
from dynamo.preprocessing.normalization import (
    calc_sz_factor,
    normalize
)
from dynamo.preprocessing.pca import (
    pca
)
from dynamo.preprocessing.Preprocessor import (
    Preprocessor
)
from dynamo.preprocessing.QC import (
    filter_cells_by_highly_variable_genes,
    filter_cells_by_outliers as monocle_filter_cells_by_outliers,
    filter_genes_by_outliers as monocle_filter_genes_by_outliers
)
from dynamo.preprocessing.transform import (
    log1p
)
from dynamo.preprocessing.utils import (
    collapse_species_adata,
    convert2symbol
)

# Imports from MultiDynamo
from ATACseqTools import (
    tfidf_normalize
)
from MultiQC import (
    modality_basic_stats,
    modality_filter_cells_by_outliers,
    modality_filter_features_by_outliers
)

# Define a custom type for the recipe dictionary using TypedDict
ATACType = Literal['archR', 'cicero', 'muon', 'signac']
CITEType = Literal['seurat']
HiCType = Literal['periwal']
ModalityType = Literal['atac', 'cite', 'hic', 'rna']
RNAType = Literal['monocle', 'seurat', 'sctransform', 'pearson_residuals', 'monocle_pearson_residuals']

class RecipeDataType(TypedDict, total=False): # total=False allows partial dictionary to be valid
    atac: ATACType
    cite: CITEType
    hic: HiCType
    rna: RNAType


# The Multiomic Preprocessor class, MultiPreprocessor
class MultiPreprocessor(Preprocessor):
    def __init__(
            self,
            cell_cycle_score_enable:                        bool=False,
            cell_cycle_score_kwargs:                        Dict[str, Any] = {},
            collapse_species_adata_function:                Callable = collapse_species_adata,
            convert_gene_name_function:                     Callable=convert2symbol,
            filter_cells_by_highly_variable_genes_function: Callable = filter_cells_by_highly_variable_genes,
            filter_cells_by_highly_variable_genes_kwargs:   Dict[str, Any] = {},
            filter_cells_by_outliers_function:              Callable=monocle_filter_cells_by_outliers,
            filter_cells_by_outliers_kwargs:                Dict[str, Any] = {},
            filter_genes_by_outliers_function:              Callable=monocle_filter_genes_by_outliers,
            filter_genes_by_outliers_kwargs:                Dict[str, Any] = {},
            force_gene_list:                                Optional[List[str]]=None,
            gene_append_list:                               List[str] = [],
            gene_exclude_list:                              List[str] = {},
            norm_method:                                    Callable=log1p,
            norm_method_kwargs:                             Dict[str, Any] = {},
            normalize_by_cells_function:                    Callable=normalize,
            normalize_by_cells_function_kwargs:             Dict[str, Any] = {},
            normalize_selected_genes_function:              Callable=None,
            normalize_selected_genes_kwargs:                Dict[str, Any] = {},
            pca_function:                                   Callable=pca,
            pca_kwargs:                                     Dict[str, Any] = {},
            regress_out_kwargs:                             Dict[List[str], Any] = {},
            sctransform_kwargs:                             Dict[str, Any] = {},
            select_genes_function:                          Callable = select_genes_monocle,
            select_genes_kwargs:                            Dict[str, Any] = {},
            size_factor_function:                           Callable=calc_sz_factor,
            size_factor_kwargs:                             Dict[str, Any] = {}) -> None:
        super().__init__(
            collapse_species_adata_function = collapse_species_adata_function,
            convert_gene_name_function = convert_gene_name_function,
            filter_cells_by_outliers_function = filter_cells_by_outliers_function,
            filter_cells_by_outliers_kwargs = filter_cells_by_outliers_kwargs,
            filter_genes_by_outliers_function = filter_genes_by_outliers_function,
            filter_genes_by_outliers_kwargs = filter_genes_by_outliers_kwargs,
            filter_cells_by_highly_variable_genes_function = filter_cells_by_highly_variable_genes_function,
            filter_cells_by_highly_variable_genes_kwargs = filter_cells_by_highly_variable_genes_kwargs,
            normalize_by_cells_function = normalize_by_cells_function,
            normalize_by_cells_function_kwargs = normalize_by_cells_function_kwargs,
            size_factor_function = size_factor_function,
            size_factor_kwargs = size_factor_kwargs,
            select_genes_function = select_genes_function,
            select_genes_kwargs = select_genes_kwargs,
            normalize_selected_genes_function = normalize_selected_genes_function,
            normalize_selected_genes_kwargs = normalize_selected_genes_kwargs,
            norm_method = norm_method,
            norm_method_kwargs = norm_method_kwargs,
            pca_function = pca_function,
            pca_kwargs = pca_kwargs,
            gene_append_list = gene_append_list,
            gene_exclude_list = gene_exclude_list,
            force_gene_list = force_gene_list,
            sctransform_kwargs = sctransform_kwargs,
            regress_out_kwargs = regress_out_kwargs,
            cell_cycle_score_enable = cell_cycle_score_enable,
            cell_cycle_score_kwargs = cell_cycle_score_kwargs
        )

    def preprocess_atac(
            self,
            mdata:           MuData,
            recipe:          ATACType = 'muon',
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None
    ) -> None:
        if recipe == 'archR':
            self.preprocess_atac_archr(mdata,
                                       tkey=tkey,
                                       experiment_type=experiment_type)
        elif recipe == 'cicero':
            self.preprocess_atac_cicero(mdata,
                                        tkey=tkey,
                                        experiment_type=experiment_type)
        elif recipe == 'muon':
            self.preprocess_atac_muon(mdata,
                                      tkey=tkey,
                                      experiment_type=experiment_type)
        elif recipe == 'signac':
            self.preprocess_atac_signac(mdata,
                                        tkey=tkey,
                                        experiment_type=experiment_type)
        else:
            raise NotImplementedError("preprocess recipe chosen not implemented: %s" % recipe)

    def preprocess_atac_archr(
            self,
            mdata: MuData,
            tkey: Optional[str] = None,
            experiment_type: Optional[str] = None
    ) -> None:
        pass

    def preprocess_atac_cicero(
            self,
            mdata: MuData,
            tkey: Optional[str] = None,
            experiment_type: Optional[str] = None
    ) -> None:
        pass

    def preprocess_atac_muon(
            self,
            mdata:           MuData,
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None
    ) -> None:
        main_info('Running muon preprocessing pipeline for scATAC-seq data ...')
        preprocess_logger = LoggerManager.gen_logger('preprocess_atac_muon')
        preprocess_logger.log_time()

        # Standardize MuData object
        self.standardize_mdata(mdata, tkey, experiment_type)

        # Filter peaks
        modality_filter_features_by_outliers(mdata,
                                             modality='atac',
                                             quantiles=[0.01, 0.99],
                                             var_key='n_cells_by_counts')

        # Filter cells
        modality_filter_cells_by_outliers(mdata,
                                          modality='atac',
                                          quantiles=[0.01, 0.99],
                                          obs_key='n_genes_by_counts')

        modality_filter_cells_by_outliers(mdata,
                                          modality='atac',
                                          quantiles=[0.01, 0.99],
                                          obs_key='total_counts')

        # Extract chromatin accessibility and transcriptome
        atac_adata, rna_adata = mdata.mod['atac'], mdata.mod['rna']

        # ... store counts layer used for SCVI's variational autoencoders
        atac_adata.layers[MDKM.ATAC_COUNTS_LAYER] = atac_adata.X
        rna_adata.layers[MDKM.RNA_COUNTS_LAYER] = rna_adata.X

        # ... compute TF-IDF
        main_info(f'computing TF-IDF', indent_level=1)
        atac_adata = tfidf_normalize(atac_adata=atac_adata, mv_algorithm=False)

        # Normalize
        main_info(f'normalizing', indent_level=1)
        sc.pp.normalize_total(atac_adata, target_sum=1e4)
        sc.pp.log1p(atac_adata)

        # Feature selection
        main_info(f'feature selection', indent_level=1)
        sc.pp.highly_variable_genes(atac_adata, min_mean=0.05, max_mean=1.5, min_disp=0.5)
        main_info(f'identified {np.sum(atac_adata.var.highly_variable)} highly variable features', indent_level=2)

        # Store current AnnData object in raw
        atac_adata.raw = atac_adata

        # Latent sematic indexing
        main_info(f'computing latent sematic indexing', indent_level=1)
        ac.tl.lsi(atac_adata)

        # ... drop first component (size related)
        main_info(f'<insert> X_lsi key in .obsm', indent_level=2)
        atac_adata.obsm[MDKM.ATAC_OBSM_LSI_KEY] = atac_adata.obsm[MDKM.ATAC_OBSM_LSI_KEY][:, 1:]
        main_info(f'<insert> LSI key in .varm', indent_level=2)
        atac_adata.varm[MDKM.ATAC_VARM_LSI_KEY] = atac_adata.varm[MDKM.ATAC_VARM_LSI_KEY][:, 1:]
        main_info(f'<insert> [lsi][stdev] key in .uns', indent_level=2)
        atac_adata.uns['lsi']['stdev'] = atac_adata.uns['lsi']['stdev'][1:]

        # ... perhaps gratuitous deep copy
        mdata.mod['atac'] = atac_adata.copy()

        preprocess_logger.finish_progress(progress_name='preprocess_atac_muon')

    def preprocess_atac_signac(
            self,
            mdata:           MuData,
            recipe:          ATACType = 'muon',
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None
    ) -> None:
        pass

    def preprocess_cite(
            self,
            mdata:  MuData,
            recipe: CITEType
    ) -> None:
        pass

    def preprocess_hic(
            self,
            mdata:  MuData,
            recipe: HiCType
    ) -> None:
        pass

    def preprocess_mdata(
            self,
            mdata:           MuData,
            recipe_dict:     RecipeDataType = None,
            tkey:            Optional[str] = None,
            experiment_type: Optional[str] = None,
    ) -> None:
        """Preprocess the MuData object with the recipe specified.

        Args:
            mdata: An AnnData object.
            recipe_dict: The recipe used to preprocess the data. Current modalities are scATAC-seq, CITE-seq, scHi-C
                         and scRNA-seq
            tkey: the key for time information (labeling time period for the cells) in .obs. Defaults to None.
            experiment_type: the experiment type of the data. If not provided, would be inferred from the data.

        Raises:
            NotImplementedError: the recipe is invalid.
        """

        if recipe_dict is None:
            # Default recipe
            recipe_dict = {'atac': 'signac', 'rna': 'seurat'}

        for mod, recipe in recipe_dict.items():
            if mod not in mdata.mod:
                main_exception((f'Modality {mod} not found in MuData object'))

            if mod == 'atac':
                self.preprocess_atac(mdata=mdata,
                                     recipe=recipe,
                                     tkey=tkey,
                                     experiment_type=experiment_type)

            elif mod == 'cite':
                self.preprocess_cite(mdata=mdata,
                                     recipe=recipe,
                                     tkey=tkey,
                                     experiment_type=experiment_type)
            elif mod == 'hic':
                self.preprocess_hic(mdata=mdata,
                                    recipe=recipe,
                                    tkey=tkey,
                                    experiment_type=experiment_type)
            elif mod == 'rna':
                rna_adata = mdata.mod.get('rna', None)

                self.preprocess_adata(adata=rna_adata,
                                      recipe=recipe,
                                      tkey=tkey,
                                      experiment_type=experiment_type)
            else:
                raise NotImplementedError(f'Preprocess recipe not implemented for modality: {mod}')

        # Integrate modalities - at this point have filtered out poor quality cells for individual
        # modalities.  Next we need to

    def standardize_mdata(
            self,
            mdata:           MuData,
            tkey:            str,
            experiment_type: str
    ) -> None:
        """Process the scATAC-seq modality within MuData to make it meet the standards of dynamo.

        The index of the observations would be ensured to be unique. The layers with sparse matrix would be converted to
        compressed csr_matrix. MDKM.allowed_layer_raw_names() will be used to define only_splicing, only_labeling and
        splicing_labeling keys.

        Args:
            mdata: an AnnData object.
            tkey: the key for time information (labeling time period for the cells) in .obs.
            experiment_type: the experiment type.
        """

        for modality, modality_adata in mdata.mod.items():
            if modality == 'rna':
                # Handled by dynamo
                continue

            # Compute basic QC metrics
            modality_basic_stats(mdata=mdata, modality=modality)

            self.add_experiment_info(modality_adata, tkey, experiment_type)
            main_info_insert_adata("tkey=%s" % tkey, "uns['pp']", indent_level=2)
            main_info_insert_adata("experiment_type=%s" % modality_adata.uns["pp"]["experiment_type"],
                                   "uns['pp']",
                                   indent_level=2)

            self.convert_layers2csr(modality_adata)
