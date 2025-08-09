# baltimoreparcel/scripts/spatial_analysis_pipeline.py
"""
Unified pipeline for spatial analysis (bivariate and Moran's I) on Baltimore property data.
"""
import sys
import time
import arcpy
import shutil
import pandas as pd
from enum import Enum
import geopandas as gpd
from pathlib import Path
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from baltimoreparcel.utils import Logger, info, warn, error, success
from datetime import datetime
from baltimoreparcel.directories import LOGS_DIR, GBD_DIR, DATA_DIR, FIGS_DIR
from baltimoreparcel.gis_utils import arcstr
from baltimoreparcel.config import ALL_YEARS
from baltimoreparcel.scripts.merge_time_layers import merge_layers
from baltimoreparcel.scripts.eda.plots import plot_line
# Configuration
arcpy.env.overwriteOutput = True
arcpy.env.workspace = arcstr(GBD_DIR)

class AnalysisType(Enum):
    BIVARIATE = "bivariate"
    MORANS_I = "morans_i"

@dataclass
class NeighborhoodSetting:
    """Configuration for spatial neighborhood definition."""
    type: str
    param: Optional[int] = None
    
    def get_suffix(self) -> str:
        """Generate suffix for output naming."""
        if self.param:
            suffix = f"{self.type.lower()}{self.param}"
        else:
            suffix = self.type.lower()
        return suffix.replace("_k_nearest_neighbors", "_knn")
    
@dataclass
class AnalysisConfig:
    """Configuration for spatial analysis."""
    lag_panel_name: str = "base_lag_panel"
    base_var: str = "LOG_REAL_NFMTTLVL_CHNG"
    dependent_vars: List[str] = None
    neighborhood_settings: List[NeighborhoodSetting] = None
    years: List[int] = None
    num_permutations: int = 199
    save_report_dir: Path = DATA_DIR / "spatial_analysis_reports"
    
    def __post_init__(self):
        if self.dependent_vars is None:
            self.dependent_vars = [
                "LOG_REAL_NFMIMPVL_CHNG", 
                "ZONING_CHNG", 
                "LOG_REAL_NFMLNDVL_CHNG"
            ]
        if self.neighborhood_settings is None:
            self.neighborhood_settings = [
                NeighborhoodSetting("K_NEAREST_NEIGHBORS", 50)
            ]
        if self.years is None:
            self.years = ALL_YEARS

        self.save_report_dir.mkdir(exist_ok=True, parents=True)

#Namespace of utility functions for spatial analysis
class SpatialAnalysisUtils:
    """Utility functions for spatial analysis pipeline."""
    
    @staticmethod
    def short_name(var: str) -> str:
        """Convert long variable names to short abbreviations."""
        mapping = {
            "LOG_REAL_NFMTTLVL_CHNG": "TTLCHG",
            "LOG_REAL_NFMIMPVL_CHNG": "IMPCHG",
            "LOG_REAL_NFMLNDVL_CHNG": "LNDCHG",
            "ZONING_CHNG": "ZCHG"
        }
        return mapping.get(var, var[:6])
    
    @staticmethod
    def safe_delete(out_path: str, max_attempts: int = 3) -> bool:
        """Safely delete ArcGIS feature class with retry logic."""
        if not arcpy.Exists(out_path):
            return True
            
        for attempt in range(1, max_attempts + 1):
            try:
                arcpy.management.Delete(out_path)
                return True
            except Exception as e:
                warn(f"Attempt {attempt}: Failed to delete {out_path}: {e}")
                if attempt < max_attempts:
                    time.sleep(2)
        
        error(f"Could not delete locked feature class: {out_path}")
        return False
    
    @staticmethod
    def get_lag_year(year: int, base_var: str, df: gpd.GeoDataFrame) -> Optional[int]:
        """Find the most recent lag year < year where base_var column exists."""
        potential_lags = sorted([
            int(col.split('_')[-1]) 
            for col in df.columns 
            if col.startswith(base_var) and col.split('_')[-1].isdigit()
        ], reverse=True)
        
        for lag in potential_lags:
            if lag < year:
                return lag
        return None
    
    @staticmethod
    def generate_output_name(analysis_type: AnalysisType, var1: str, var2: str = None, 
                           nb_setting: str = "") -> str:
        """Generate standardized output name."""
        prefix = analysis_type.value
        v1 = SpatialAnalysisUtils.short_name(var1)
        
        if var2:  # Bivariate
            v2 = SpatialAnalysisUtils.short_name(var2)
            return f"{prefix}_{v1}_{v2}_{nb_setting}"
        else:  # Univariate (Moran's I)
            return f"{prefix}_{v1}_{nb_setting}"

#Namespace of functions for cleaning and standardization for spatial analysis outputs
class FieldCleaner:
    """Handles field cleaning and standardization for spatial analysis outputs."""
    
    # Field mappings for different analysis types
    BIVARIATE_FIELD_MAP = {
        'LOCAL_L': 'BIVAR_STAT',
        'P_VALUE': 'BIVAR_PVALUE',
        'SIG_LEVEL': 'SIGNIF_LEVEL', 
        'ASSOC_CAT': 'CLUSTER_TYPE',
        'NUM_NBRS': 'N_NEIGHBORS'
    }
    
    MORANS_FIELD_MAP = {
        'Gi_Bin': 'SIGNIF_BIN',
        'GiZScore': 'MORAN_ZSCORE',
        'GiPValue': 'MORAN_PVALUE'
    }
    
    @staticmethod
    def get_essential_fields(analysis_type: AnalysisType) -> List[str]:
        """Get list of essential fields to keep based on analysis type."""
        base_fields = [
            'OBJECTID', 'Shape',
            'ACCTID', 'GEOGCODE', 'NEIGHBORHOOD', 'SOURCE_ID',
            'ANALYSIS_YEAR', 'LAG_YEAR', 'NB_SETTING', 'RESPONSE_VAR'
        ]
        
        if analysis_type == AnalysisType.BIVARIATE:
            base_fields.extend([
                'LOCAL_L', 'P_VALUE', 'SIG_LEVEL', 'ASSOC_CAT', 'NUM_NBRS', 'EXPLAN_VAR'
            ])
        elif analysis_type == AnalysisType.MORANS_I:
            base_fields.extend([
                'Gi_Bin', 'GiZScore', 'GiPValue'  # Adjust based on actual Moran's I output
            ])
        
        return base_fields
    
    @staticmethod
    def get_field_mapping(analysis_type: AnalysisType) -> Dict[str, str]:
        """Get field renaming mapping based on analysis type."""
        if analysis_type == AnalysisType.BIVARIATE:
            return FieldCleaner.BIVARIATE_FIELD_MAP
        elif analysis_type == AnalysisType.MORANS_I:
            return FieldCleaner.MORANS_FIELD_MAP
        return {}
    
class SpatialAnalyzer:
    """Main class for running spatial analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.lag_panel_path = arcstr(GBD_DIR / config.lag_panel_name)
        
    def load_lag_panel(self) -> gpd.GeoDataFrame:
        """Load the lag panel data."""
        if not arcpy.Exists(self.lag_panel_path):
            raise FileNotFoundError(f"Lag panel '{self.lag_panel_path}' does not exist.")
        
        info(f"Reading lag panel from {self.lag_panel_path}...")
        gdf = gpd.read_file(str(GBD_DIR), layer=self.config.lag_panel_name)
        info(f"Lag panel shape: {gdf.shape}")
        return gdf
        
    def run_spatial_analysis(self, analysis_type: AnalysisType, 
                           analysis_field1: str, analysis_field2: str = None,
                           year: int = None, lag_year: int = None,
                           nb_setting: NeighborhoodSetting = None,
                           output_name: str = None) -> Optional[str]:
        """Run spatial analysis (bivariate or Moran's I)."""
        
        if analysis_type == AnalysisType.MORANS_I:
            # Moran's I doesn't create feature classes, just run the analysis
            try:
                self._run_morans_analysis(analysis_field1, None, nb_setting)
                success(f"Completed Moran's I analysis for {analysis_field1}")
                return f"morans_{analysis_field1}_{nb_setting.get_suffix()}"
            except Exception as e:
                error(f"Failed Moran's I analysis: {e}. Returning None.")
                return None
            
        if not output_name:
            base_name = SpatialAnalysisUtils.generate_output_name(
                analysis_type, analysis_field1, analysis_field2, nb_setting.get_suffix()
            )
            output_name = f"{base_name}_{year}_{lag_year}" if lag_year else f"{base_name}_{year}"
        
        out_path = arcstr(GBD_DIR / output_name)
        
        if not SpatialAnalysisUtils.safe_delete(out_path):
            return None
            
        try:
            if analysis_type == AnalysisType.BIVARIATE:
                self._run_bivariate_analysis(
                    analysis_field1, analysis_field2, out_path, nb_setting
                )
                        
            # Clean the output
            self._clean_output(
                out_path, analysis_type, analysis_field1, analysis_field2,
                year, lag_year, nb_setting.get_suffix()
            )
            
            success(f"Completed {analysis_type.value} analysis: {output_name}")
            return output_name
            
        except Exception as e:
            error(f"Failed {analysis_type.value} analysis: {e}")
            return None
    
    def _run_bivariate_analysis(self, field1: str, field2: str, out_path: str,
                               nb_setting: NeighborhoodSetting):
        """Run bivariate spatial association analysis."""
        arcpy.stats.BivariateSpatialAssociation(
            in_features=self.lag_panel_path,
            analysis_field1=field1,
            analysis_field2=field2,
            out_features=out_path,
            neighborhood_type=nb_setting.type,
            num_neighbors=nb_setting.param if nb_setting.type == "K_NEAREST_NEIGHBORS" else None,
            local_weighting_scheme="UNWEIGHTED",
            num_permutations=self.config.num_permutations
        )
    
    def _run_morans_analysis(self, field: str, out_path: str,
                           nb_setting: NeighborhoodSetting) -> Dict[str, float]:
        """Run Moran's I spatial autocorrelation analysis."""
        try:
            # Run Global Moran's I
            result = arcpy.stats.SpatialAutocorrelation(
                Input_Feature_Class=self.lag_panel_path,
                Input_Field=field,
                Generate_Report="GENERATE_REPORT",
                Conceptualization_of_Spatial_Relationships=self._get_spatial_relationship(nb_setting),
                Distance_Method="EUCLIDEAN_DISTANCE",
                Standardization="ROW",
                Distance_Band_or_Threshold_Distance=self._get_distance_threshold(nb_setting),
                Weights_Matrix_File=None,
                number_of_neighbors=nb_setting.param if nb_setting.type == "K_NEAREST_NEIGHBORS" else None
            )
        except:
            print(f"failed the Moran's I analysis")

        print('SpatialAutocorrelation task complete.')
        print(f"{result=}")

        try:    
            # Extract results from the correct output indices
            morans_results = {
                'morans_i': float(result.getOutput(0)),      # Moran's I statistic
                'z_score': float(result.getOutput(1)),       # Z-score  
                'p_value': float(result.getOutput(2)),       # P-value
                'html_report': str(result.getOutput(3)),     # HTML report path
                'input_field': str(result.getOutput(4))      # Input field name
            }
            
            # Calculate additional derived statistics
            # Expected I for n observations is -1/(n-1)
            n_features = int(arcpy.management.GetCount(self.lag_panel_path)[0])
            morans_results['expected_i'] = -1.0 / (n_features - 1)
            
            print("Copying html report")
            shutil.copy(morans_results['html_report'], self.config.save_report_dir / f"morans_i_report_{field}.html")
            # Calculate variance from z-score and Moran's I if possible
            if morans_results['z_score'] != 0:
                morans_results['variance'] = ((morans_results['morans_i'] - morans_results['expected_i']) / morans_results['z_score']) ** 2
            else:
                morans_results['variance'] = None
            
            info(f"Global Moran's I Results for {field}:")

            return morans_results
            
        except Exception as e:
            error(f"Error extracting Moran's I results: {e}")
            # Fallback: try to extract from result messages
            return self._extract_morans_from_messages(result)
        
    def _extract_morans_from_messages(self, result) -> Dict[str, float]:
        """Fallback method to extract Moran's I results from result messages."""
        try:
            messages = result.getMessages()
            morans_results = {}
            
            # Parse the messages for key statistics
            for message in messages.split('\n'):
                if 'Moran\'s Index:' in message:
                    morans_results['morans_i'] = float(message.split(':')[1].strip())
                elif 'Expected Index:' in message:
                    morans_results['expected_i'] = float(message.split(':')[1].strip())
                elif 'Variance:' in message:
                    morans_results['variance'] = float(message.split(':')[1].strip())
                elif 'z-score:' in message:
                    morans_results['z_score'] = float(message.split(':')[1].strip())
                elif 'p-value:' in message:
                    morans_results['p_value'] = float(message.split(':')[1].strip())
            
            return morans_results
            
        except Exception as e:
            error(f"Could not extract Moran's I results from messages: {e}")
            return {
                'morans_i': None,
                'expected_i': None,
                'variance': None,
                'z_score': None,
                'p_value': None
            }
    
    def _get_spatial_relationship(self, nb_setting: NeighborhoodSetting) -> str:
        """Convert neighborhood setting to spatial relationship type."""
        if nb_setting.type == "K_NEAREST_NEIGHBORS":
            return "K_NEAREST_NEIGHBORS"
        elif nb_setting.type == "DISTANCE_BAND":
            return "INVERSE_DISTANCE"
        elif "CONTIGUITY" in nb_setting.type:
            return "CONTIGUITY_EDGES_ONLY" if "EDGES_ONLY" in nb_setting.type else "CONTIGUITY_EDGES_CORNERS"
        else:
            return "INVERSE_DISTANCE"  # Default
    
    def _get_distance_threshold(self, nb_setting: NeighborhoodSetting) -> Optional[float]:
        """Get distance threshold based on neighborhood setting."""
        if nb_setting.type == "DISTANCE_BAND":
            return nb_setting.param
        elif nb_setting.type == "K_NEAREST_NEIGHBORS":
            return None  # Uses number_of_neighbors instead
        else:
            return 200  # Default from your example
        
    def _clean_output(self, out_path: str, analysis_type: AnalysisType,
                     field1: str, field2: str = None, year: int = None,
                     lag_year: int = None, nb_setting: str = ""):
        """Clean and standardize output feature class."""
        
        temp_join_fc = out_path + "_temp_join"
        
        try:
            # Step 1: Spatial join to recover identifiers
            if arcpy.Exists(temp_join_fc):
                arcpy.management.Delete(temp_join_fc)
            
            info("Performing spatial join to recover parcel identifiers...")
            arcpy.analysis.SpatialJoin(
                target_features=out_path,
                join_features=self.lag_panel_path,
                out_feature_class=temp_join_fc,
                join_operation="JOIN_ONE_TO_ONE",
                join_type="KEEP_ALL",
                match_option="INTERSECT"
            )
            
            # Step 2: Add metadata fields
            self._add_metadata_fields(temp_join_fc, field1, field2, year, lag_year, nb_setting)
            
            # Step 3: Remove unnecessary fields
            self._remove_unnecessary_fields(temp_join_fc, analysis_type)
            
            # Step 4: Rename fields for clarity
            self._rename_fields(temp_join_fc, analysis_type)
            
            # Step 5: Replace original with cleaned version
            if arcpy.Exists(out_path):
                arcpy.management.Delete(out_path)
            arcpy.management.Rename(temp_join_fc, out_path)
            
            # Step 6: Summary
            count = int(arcpy.management.GetCount(out_path)[0])
            fields = [f.name for f in arcpy.ListFields(out_path)]
            info(f"Cleaned FC: {count} records, {len(fields)} fields")
            
        except Exception as e:
            error(f"Failed to clean output {out_path}: {e}")
            if arcpy.Exists(temp_join_fc):
                try:
                    arcpy.management.Delete(temp_join_fc)
                except:
                    pass
            raise
    
    def _add_metadata_fields(self, fc_path: str, field1: str, field2: str = None,
                           year: int = None, lag_year: int = None, nb_setting: str = ""):
        """Add and populate metadata fields."""
        
        # Define metadata fields based on analysis type
        metadata_fields = [
            ("ANALYSIS_YEAR", "SHORT"),
            ("LAG_YEAR", "SHORT"), 
            ("NB_SETTING", "TEXT", 50),
            ("RESPONSE_VAR", "TEXT", 50)
        ]
        
        if field2:  # Bivariate analysis
            metadata_fields.append(("EXPLAN_VAR", "TEXT", 50))
        
        # Add fields
        for field_info in metadata_fields:
            field_name, field_type = field_info[0], field_info[1]
            field_length = field_info[2] if len(field_info) > 2 else None
            
            if field_length:
                arcpy.management.AddField(fc_path, field_name, field_type, field_length=field_length)
            else:
                arcpy.management.AddField(fc_path, field_name, field_type)
        
        # Populate fields
        response_var = field1.rsplit('_', 1)[0] if '_' in field1 else field1
        explan_var = field2.rsplit('_', 1)[0] if field2 and '_' in field2 else field2
        
        cursor_fields = ["ANALYSIS_YEAR", "LAG_YEAR", "NB_SETTING", "RESPONSE_VAR"]
        cursor_values = [year, lag_year, nb_setting, response_var]
        
        if field2:
            cursor_fields.append("EXPLAN_VAR")
            cursor_values.append(explan_var)
        
        with arcpy.da.UpdateCursor(fc_path, cursor_fields) as cursor:
            for row in cursor:
                for i, val in enumerate(cursor_values):
                    row[i] = val
                cursor.updateRow(row)
    
    def _remove_unnecessary_fields(self, fc_path: str, analysis_type: AnalysisType):
        """Remove unnecessary fields, keeping only essential ones."""
        essential_fields = FieldCleaner.get_essential_fields(analysis_type)
        existing_fields = [f.name for f in arcpy.ListFields(fc_path)]
        
        # Protected fields that can't/shouldn't be deleted
        protected = ['OBJECTID', 'Shape', 'OBJECTID_1', 'Shape_Length', 'Shape_Area']
        
        fields_to_delete = [
            f for f in existing_fields 
            if f not in essential_fields and f not in protected
        ]
        
        if fields_to_delete:
            info(f"Removing {len(fields_to_delete)} unnecessary fields...")
            try:
                arcpy.management.DeleteField(fc_path, fields_to_delete)
            except Exception as e:
                warn(f"Could not delete some fields: {e}")
    
    def _rename_fields(self, fc_path: str, analysis_type: AnalysisType):
        """Rename fields for clarity based on analysis type."""
        field_mappings = FieldCleaner.get_field_mapping(analysis_type)
        
        for old_name, new_name in field_mappings.items():
            if any(f.name == old_name for f in arcpy.ListFields(fc_path)):
                try:
                    arcpy.management.AlterField(fc_path, old_name, new_name, new_name)
                except Exception as e:
                    warn(f"Could not rename {old_name} to {new_name}: {e}")

def run_bivariate_pipeline(config: AnalysisConfig = None) -> List[str]:
    """Default: Run the complete bivariate analysis pipeline."""
    if not config:
        config = AnalysisConfig()
    
    analyzer = SpatialAnalyzer(config)
    lag_panel_gdf = analyzer.load_lag_panel()
    lag_panel_cols = lag_panel_gdf.columns.tolist()
    
    merged_outputs = []
    
    for explan_var in config.dependent_vars:
        for nb_setting in config.neighborhood_settings:
            output_base = SpatialAnalysisUtils.generate_output_name(
                AnalysisType.BIVARIATE, config.base_var, explan_var, nb_setting.get_suffix()
            )
            
            # Run analysis for each year
            for year in config.years:
                lag_year = SpatialAnalysisUtils.get_lag_year(year, config.base_var, lag_panel_gdf)
                if not lag_year:
                    warn(f"No lag year found for {year}")
                    continue
                    
                field1 = f"{config.base_var}_{year}"
                field2 = f"{explan_var}_{lag_year}"
                
                if field1 not in lag_panel_cols or field2 not in lag_panel_cols:
                    warn(f"Missing fields for {year}: {field1}, {field2}")
                    continue
                
                analyzer.run_spatial_analysis(
                    AnalysisType.BIVARIATE, field1, field2, 
                    year, lag_year, nb_setting
                )
            
            # Merge yearly results
            try:
                merged_name = f"{output_base}_merged"
                merge_layers(prefix=output_base, output_fc=merged_name, convert_time=True)
                merged_outputs.append(merged_name)
                if len(merged_outputs) > 0:
                    success(f"Merged {len(merged_outputs)} layers into {merged_name}")

            except Exception as e:
                error(f"Failed to merge {output_base}: {e}")
    
    return merged_outputs

def run_morans_pipeline(config: AnalysisConfig = None) -> List[str]:
    """Run the complete Moran's I analysis pipeline."""
    if not config:
        config = AnalysisConfig()
    
    analyzer = SpatialAnalyzer(config)
    analysis_type = AnalysisType.MORANS_I
    # Load lag panel
    info("Loading lag panel for Moran's I analysis...")
    lag_panel_gdf = analyzer.load_lag_panel()
    lag_panel_cols = lag_panel_gdf.columns.tolist()
    
    results = []
    morans_data = []  # Store results for DataFrame
    
    for nb_setting in config.neighborhood_settings:
        output_base = SpatialAnalysisUtils.generate_output_name(
            analysis_type, config.base_var, None, nb_setting.get_suffix()
        )
        # Run analysis for each year
        for year in config.years:
            print("-"*100)
            print(f"Processing Moran's I for year: {year} with settings: {nb_setting.get_suffix()}")
            field = f"{config.base_var}_{year}"
            
            if field not in lag_panel_cols:
                warn(f"Missing field for {year}: {field}")
                continue
            
            try:
                info(f"Running Moran's I for {field} ({nb_setting.get_suffix()})...")
                morans_results = analyzer._run_morans_analysis(field, None, nb_setting)
                success(f"Moran's I analysis completed:\n{morans_results}")
                # Store results with metadata
                result_record = {
                    'year': year,
                    'field': field,
                    'neighborhood_setting': nb_setting.get_suffix(),
                    'neighborhood_type': nb_setting.type,
                    'neighborhood_param': nb_setting.param,
                    'base_var': config.base_var,
                    **morans_results  # Unpacks morans_i, z_score, p_value, etc.
                }
                morans_data.append(result_record)
                
                success(f"Completed Moran's I for {year}")
                results.append(f"{output_base}_{year}")
                
            except Exception as e:
                error(f"Failed Moran's I analysis for {year}: {e}")
    
    # Create DataFrame with all results
    results_df = pd.DataFrame(morans_data)
    csv_path = None
    # Save results to CSV
    if not results_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = DATA_DIR / f"morans_i_results_{config.base_var}_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        success(f"Saved Moran's I results to {csv_path}")
        
        # Print summary
        info(f"Moran's I Summary:")
        info(f"  Total analyses: {len(results_df)}")
        info(f"  Years: {sorted(results_df['year'].unique())}")
        info(f"  Neighborhood settings: {results_df['neighborhood_setting'].unique().tolist()}")
        
        # Show significant results
        if 'p_value' in results_df.columns:
            significant = results_df[results_df['p_value'] < 0.05]
            info(f"  Significant results (p < 0.05): {len(significant)}")


    info(f"Moran's I analysis completed for {len(results)} year-setting combinations")
    
    return results, results_df, csv_path


if __name__ == "__main__":
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = LOGS_DIR / f"spatial_analysis_{timestamp}.log"
    logger = Logger(log_file)
    
    # Execution flags
    execute_bivariate_pipeline = False
    execute_morans_pipeline = True
    
    # Configure bivariate analysis
    bivariate_config = AnalysisConfig(
        lag_panel_name="base_lag_panel",#layer being used in GBD_DIR
        base_var="LOG_REAL_NFMTTLVL_CHNG",
        dependent_vars=[
            "LOG_REAL_NFMIMPVL_CHNG", 
            "LOG_REAL_NFMLNDVL_CHNG",
            "ZONING_CHNG", 
        ],
        neighborhood_settings=[
            # NeighborhoodSetting("K_NEAREST_NEIGHBORS", 20),
            NeighborhoodSetting("K_NEAREST_NEIGHBORS", 50),
            # NeighborhoodSetting("K_NEAREST_NEIGHBORS", 100)
        ],
        years=ALL_YEARS,  # Use all available years
        num_permutations=199
    )
    
    # Configure Moran's I analysis
    morans_i_config = AnalysisConfig(
        lag_panel_name="base_lag_panel",
        # lag_panel_name="biv_TTLCHG_IMPCHG_knn50_merged_panel",
        # lag_panel_name="LOG_REAL_NFMTTLVL_panel",
        # lag_panel_name="LOG_REAL_NFMIMPVL_panel",
        # base_var="BIVAR_PVALUE",
        # base_var="LOG_REAL_NFMTTLVL_CHNG",
        base_var="LOG_REAL_NFMIMPVL_CHNG",
        # base_var="LOG_REAL_NFMIMPVL",
        dependent_vars=None,  # Not used for Moran's I
        neighborhood_settings=[
            NeighborhoodSetting("K_NEAREST_NEIGHBORS", 50),
            # NeighborhoodSetting("DISTANCE_BAND", 200)
        ],
        years=ALL_YEARS,
        num_permutations=999  # More permutations for Moran's I
    )
    
    ##################################
    # Run analyses based on flags
    ##################################
    # BIVARIATE ANALYSIS PIPELINE
    if execute_bivariate_pipeline:
        info("Starting bivariate analysis pipeline...")
        info(f"Config: {len(bivariate_config.dependent_vars)} dependent vars, "
             f"{len(bivariate_config.neighborhood_settings)} neighborhood settings, "
             f"{len(bivariate_config.years)} years")
        bivariate_outputs = run_bivariate_pipeline(bivariate_config)
        biv_output_str = f"Bivariate: {len(bivariate_outputs)} merged outputs"
    else:
        biv_output_str = "Bivariate analysis skipped"
    
    ##################################
    # MORAN'S I ANALYSIS PIPELINE
    if execute_morans_pipeline:
        info("Starting Moran's I analysis pipeline...")
        info(f"Config: {len(morans_i_config.neighborhood_settings)} neighborhood settings, "
             f"{len(morans_i_config.years)} years, "
             f"{morans_i_config.num_permutations} permutations")
        morans_outputs, morans_df, morans_I_data_path = run_morans_pipeline(morans_i_config)
        morans_output_str = f"Moran's I: {len(morans_outputs)} analyses, results saved to {morans_I_data_path.name}"
        
        # Optional: Print some key results
        if not morans_df.empty and 'morans_i' in morans_df.columns:
            avg_morans_i = morans_df['morans_i'].mean()
            info(f"Average Moran's I across all analyses: {avg_morans_i:.6f}")
        else:
            error("No valid Moran's I results found.")
    else:
        morans_output_str = "Moran's I analysis skipped"

    print("\n", '-'*80)
    success(f"Analysis completed.\n{biv_output_str}\n{morans_output_str}")