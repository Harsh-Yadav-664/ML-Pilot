import pandas as pd
import numpy as np

class DataCleanAdapter:
    """Bulletproof Data Cleaning Adapter for MLPilot."""

    def clean(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Cleans the dataframe according to basic robustness rules."""
        try:
            # Make a copy so we don't modify the original dataframe unexpectedly
            df_clean = df.copy()
            
            # 1. Drop columns where more than 60% of values are missing
            missing_threshold = 0.6
            missing_fractions = df_clean.isnull().mean()
            cols_to_drop = missing_fractions[missing_fractions > missing_threshold].index.tolist()
            # Do not drop target column even if missing > 60%
            if target_column in cols_to_drop:
                cols_to_drop.remove(target_column)
            df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # 2. Drop ID-like columns (nunique == nrows), except target
            nrows = len(df_clean)
            id_cols = [c for c in df_clean.columns if c != target_column and df_clean[c].nunique() == nrows]
            df_clean.drop(columns=id_cols, inplace=True, errors='ignore')

            # Separate numeric and categorical columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

            # Remove target column from imputations if present
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            if target_column in cat_cols:
                cat_cols.remove(target_column)

            # 3. For numeric columns with < 60% missing: impute with median
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    # If entire column is NaN, median is NaN, fallback to 0
                    if pd.isna(median_val):
                        median_val = 0
                    df_clean[col].fillna(median_val, inplace=True)

            # 4. For categorical columns with < 60% missing: impute with mode
            for col in cat_cols:
                if df_clean[col].isnull().any():
                    mode_series = df_clean[col].mode()
                    if not mode_series.empty:
                        mode_val = mode_series.iloc[0]
                    else:
                        mode_val = "missing"
                    df_clean[col].fillna(mode_val, inplace=True)

            # 5. Converts all object dtype columns to pd.Categorical codes
            # (Re-fetch categorical columns in case types changed or some were missed)
            object_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
            if target_column in object_cols:
                object_cols.remove(target_column)
                
            for col in object_cols:
                df_clean[col] = df_clean[col].astype('category').cat.codes

            return df_clean
        except Exception as e:
            # Must NEVER crash
            print(f"DataCleanAdapter encountered an error: {e}")
            return df
