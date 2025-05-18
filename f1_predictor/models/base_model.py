import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import warnings
import os
from abc import ABC, abstractmethod # For abstract base class

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define data directory at the module level or pass to class
DATA_DIR_GLOBAL = "test_output/dataset"

class AbstractBasePredictor(ABC):
    def __init__(self, data_dir=DATA_DIR_GLOBAL, test_size_ratio=0.20):
        self.data_dir = data_dir
        self.test_size_ratio = test_size_ratio
        self.model = None
        self.evals_result = {}
        self.features_df = None
        self.labels_df = None
        self.merged_df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_set_full_info_df = None
        self.test_races_identifiers = None
        self.feature_columns = []
        self.target_column = 'final_position' # Default target

    def _get_latest_data_files(self):
        """Get the most recent feature and label files from the data directory."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        files = os.listdir(self.data_dir)
        feature_files = sorted([f for f in files if f.startswith('features_') and f.endswith('.csv')])
        label_files = sorted([f for f in files if f.startswith('labels_') and f.endswith('.csv')])

        if not feature_files:
            raise FileNotFoundError(f"No files starting with 'features_' and ending with '.csv' found in {self.data_dir}")
        if not label_files:
            raise FileNotFoundError(f"No files starting with 'labels_' and ending with '.csv' found in {self.data_dir}")

        latest_features_file_name = feature_files[-1]
        try:
            feature_timestamp_str = latest_features_file_name.split('_', 1)[1].rsplit('.', 1)[0]
            expected_label_file_name = f"labels_{feature_timestamp_str}.csv"
            if expected_label_file_name in label_files:
                latest_labels_file_name = expected_label_file_name
            else:
                print(f"Warning: Label file '{expected_label_file_name}' not found for features '{latest_features_file_name}'. Using alphabetically last: '{label_files[-1]}'.")
                latest_labels_file_name = label_files[-1]
        except IndexError:
            print(f"Warning: Could not parse timestamp from '{latest_features_file_name}'. Using last label file: '{label_files[-1]}'.")
            latest_labels_file_name = label_files[-1]
            
        return os.path.join(self.data_dir, latest_features_file_name), os.path.join(self.data_dir, latest_labels_file_name)

    def load_and_prepare_data(self, drop_X_qualifying_cols=True): # Argument to control dropping in _preprocess_features
        """Loads, merges, and prepares data for modeling."""
        features_file, labels_file = self._get_latest_data_files()
        self.features_df = pd.read_csv(features_file)
        self.labels_df = pd.read_csv(labels_file)
        
        self.features_df.columns = self.features_df.columns.str.strip()
        self.labels_df.columns = self.labels_df.columns.str.strip()
        
        print(f"Loaded data from:\n{features_file}\n{labels_file}")
        # print(f"DEBUG LOAD: Columns in self.features_df after loading and stripping: {self.features_df.columns.tolist()}")


        for df_ in [self.features_df, self.labels_df]:
            if 'Unnamed: 0' in df_.columns:
                df_.drop(columns=['Unnamed: 0'], inplace=True)

        self.merged_df = pd.merge(self.features_df, self.labels_df, on=['driver_id', 'season', 'race'], how='inner')
        print(f"\nShape of merged_df: {self.merged_df.shape}")
        # print(f"DEBUG MERGE: Columns in self.merged_df after merge: {self.merged_df.columns.tolist()}")


        if self.merged_df[self.target_column].isnull().any():
            print(f"Warning: NaNs found in target '{self.target_column}'. Dropping rows.")
            self.merged_df.dropna(subset=[self.target_column], inplace=True)

        self.y = self.merged_df[self.target_column]
        excluded_cols_from_X = [self.target_column, 'driver_id', 'final_time', 'delta_pos', 'season', 'race']
        
        potential_feature_cols = [col for col in self.merged_df.columns if col not in excluded_cols_from_X]
        self.feature_columns = [col for col in potential_feature_cols if col in self.features_df.columns]


        self.X = self.merged_df[self.feature_columns].copy()
        print(f"\nShape of X (features): {self.X.shape}")
        # print(f"DEBUG X_SELECT: Columns selected for X (self.feature_columns): {self.feature_columns}")

        # Pass the argument to _preprocess_features
        self._preprocess_features(drop_qualifying_cols=drop_X_qualifying_cols)
        self._split_data_chronological()

    def _preprocess_features(self, drop_qualifying_cols=True): # Added argument with default
        """Converts data types. Optionally drops qualifying time columns. NaNs are left for the model to handle by default."""
        qualifying_time_cols = ['qualifying_time_q1', 'qualifying_time_q2', 'qualifying_time_q3', 'gap_to_pole_seconds', 'championship_points', 'team_championship_position', 'season_wins']
        cols_to_drop_from_X = []
  
        for col in qualifying_time_cols:
            if col in self.X.columns:
                self.X[col] = pd.to_numeric(self.X[col], errors='coerce')
                if drop_qualifying_cols:
                    cols_to_drop_from_X.append(col)
        
        if drop_qualifying_cols and cols_to_drop_from_X:
            self.X.drop(columns=cols_to_drop_from_X, inplace=True)
            print(f"\nDropped qualifying time columns from X: {cols_to_drop_from_X}")
            # Update self.feature_columns as well
            self.feature_columns = [col for col in self.feature_columns if col not in cols_to_drop_from_X]


        for col in self.X.columns: # Iterate over remaining columns in X
            if self.X[col].dtype == 'object':
                try:
                    self.X[col] = pd.to_numeric(self.X[col], errors='coerce')
                    print(f"Column '{col}' was object, converted to numeric.")
                except ValueError:
                    print(f"Could not convert column '{col}' to numeric. NaNs will be handled by the model.")
        
        print("\n--- NaN counts in X (to be handled by model) ---")
        nan_counts = self.X.isnull().sum()
        print(nan_counts[nan_counts > 0])

    def _split_data_chronological(self):
        """Splits data chronologically based on unique races."""
        merged_df_sorted = self.merged_df.sort_values(by=['season', 'race', 'driver_id'])
        X_sorted = self.X.loc[merged_df_sorted.index]
        y_sorted = self.y.loc[merged_df_sorted.index]

        unique_races_df = merged_df_sorted[['season', 'race']].drop_duplicates().sort_values(by=['season', 'race'])
        print(f"\nTotal unique races in dataset: {len(unique_races_df)}")

        if len(unique_races_df) < 2:
            print("Not enough unique races for chronological split. Using random split as fallback.")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_sorted, y_sorted, test_size=0.25, random_state=42)
            
            columns_for_test_info = ['season', 'race', 'driver_id']
            if 'grid_position' in self.merged_df.columns: 
                columns_for_test_info.append('grid_position')
            self.test_set_full_info_df = self.merged_df.loc[self.X_test.index, columns_for_test_info].copy()
            
            if 'driver_name' in self.features_df.columns:
                driver_names_map = self.features_df[['driver_id', 'driver_name']].drop_duplicates().set_index('driver_id')['driver_name']
                self.test_set_full_info_df['driver_name'] = self.test_set_full_info_df['driver_id'].map(driver_names_map).fillna(self.test_set_full_info_df['driver_id'])
            else:
                self.test_set_full_info_df['driver_name'] = self.test_set_full_info_df['driver_id']
            self.test_races_identifiers = pd.DataFrame()
            return

        num_total_races = len(unique_races_df)
        num_test_races = max(1, int(np.ceil(num_total_races * self.test_size_ratio)))
        num_train_races = num_total_races - num_test_races

        if num_train_races == 0:
            if num_total_races > 1: num_train_races = 1; num_test_races = num_total_races - 1
            else: raise ValueError("Only 1 unique race available. Cannot perform train/test split.")

        self.test_races_identifiers = unique_races_df.tail(num_test_races)
        train_races_identifiers = unique_races_df.head(num_train_races)
        
        print(f"\nNumber of training races: {num_train_races}")
        print(f"Number of testing races: {num_test_races}")
        print("\nTest races identifiers:")
        print(self.test_races_identifiers)

        train_indices_mask = merged_df_sorted.apply(lambda row: (row['season'], row['race']) in train_races_identifiers.itertuples(index=False, name=None), axis=1)
        test_indices_mask = merged_df_sorted.apply(lambda row: (row['season'], row['race']) in self.test_races_identifiers.itertuples(index=False, name=None), axis=1)

        self.X_train = X_sorted[train_indices_mask]
        self.y_train = y_sorted[train_indices_mask]
        self.X_test = X_sorted[test_indices_mask]
        self.y_test = y_sorted[test_indices_mask]
        
        columns_for_test_info = ['season', 'race', 'driver_id']
        #print(f"DEBUG SPLIT: Columns in merged_df_sorted before check: {merged_df_sorted.columns.tolist()}")
        if 'grid_position' in merged_df_sorted.columns: 
            # print("DEBUG SPLIT: 'grid_position' IS in merged_df_sorted.columns for test_info")
            columns_for_test_info.append('grid_position')
        else:
            print("DEBUG SPLIT: 'grid_position' IS NOT in merged_df_sorted.columns for test_info")
            
        self.test_set_full_info_df = merged_df_sorted.loc[test_indices_mask, columns_for_test_info].copy()
        #print(f"DEBUG SPLIT: Columns in self.test_set_full_info_df after creation: {self.test_set_full_info_df.columns.tolist()}")
        
        if 'driver_name' in self.features_df.columns: 
            driver_names_map = self.features_df[['driver_id', 'driver_name']].drop_duplicates().set_index('driver_id')['driver_name']
            self.test_set_full_info_df['driver_name'] = self.test_set_full_info_df['driver_id'].map(driver_names_map).fillna(self.test_set_full_info_df['driver_id'])
        elif 'driver_name' in merged_df_sorted.columns: 
             self.test_set_full_info_df['driver_name'] = merged_df_sorted.loc[test_indices_mask, 'driver_name']
        else: 
            self.test_set_full_info_df['driver_name'] = self.test_set_full_info_df['driver_id']


        print(f"\nTraining set shape: X_train {self.X_train.shape}, y_train {self.y_train.shape}")
        print(f"Testing set shape: X_test {self.X_test.shape}, y_test {self.y_test.shape}")

        if self.X_train.empty or self.X_test.empty:
            raise ValueError("Training or testing set is empty after splitting.")

    @abstractmethod
    def _get_default_model_params(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self, X_data):
        pass
        
    @abstractmethod
    def get_feature_importances(self):
        pass

    def evaluate_model(self):
        """Evaluates the trained model and prints metrics. Relies on self.predict()."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        print("\n--- Evaluating model ---")
        y_pred_train = self.predict(self.X_train)
        y_pred_test = self.predict(self.X_test)

        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)

        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Train R2 Score: {train_r2:.4f}")
        print(f"Test R2 Score: {test_r2:.4f}")
        
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None and self.evals_result:
            print(f"\nBest iteration: {self.model.best_iteration}")
            model_eval_metrics = self.model.get_params().get('eval_metric', ["rmse"])
            primary_metric = model_eval_metrics[0] if isinstance(model_eval_metrics, list) else model_eval_metrics
            
            if primary_metric in self.evals_result.get('validation_0', {}):
                 print(f"Best Train {primary_metric.upper()}: {self.evals_result['validation_0'][primary_metric][self.model.best_iteration]:.4f}")
            if primary_metric in self.evals_result.get('validation_1', {}):
                 print(f"Best Test {primary_metric.upper()}: {self.evals_result['validation_1'][primary_metric][self.model.best_iteration]:.4f}")

        avg_spearman, avg_kendall = None, None
        if not self.test_set_full_info_df.empty:
            all_race_spearman_coeffs = []
            all_race_kendall_coeffs = []
            
            eval_df = self.test_set_full_info_df.copy()
            # Align y_test and y_pred_test with eval_df using original indices
            # This assumes self.test_set_full_info_df retains original indices from merged_df_sorted
            # and self.y_test / self.X_test also retain these original indices.
            try:
                eval_df['true_position'] = self.y_test.loc[eval_df.index].values
                # y_pred_test is a numpy array, needs to be Series with X_test index for .loc
                y_pred_test_series = pd.Series(y_pred_test, index=self.X_test.index)
                eval_df['predicted_score_for_ranking'] = y_pred_test_series.loc[eval_df.index].values
            except KeyError as e:
                print(f"KeyError during alignment in evaluate_model: {e}. This might indicate an index issue.")
                print(f"eval_df index: {eval_df.index[:5]}")
                print(f"self.y_test index: {self.y_test.index[:5]}")
                print(f"self.X_test index: {self.X_test.index[:5]}")
                return {"test_rmse": test_rmse, "test_r2": test_r2, "avg_spearman": None, "avg_kendall": None}


            for (season_val, race_val), group in eval_df.groupby(['season', 'race']):
                if 'true_position' not in group.columns or 'predicted_score_for_ranking' not in group.columns:
                    print(f"Race ({season_val}, {race_val}): Missing 'true_position' or 'predicted_score_for_ranking'. Skipping rank metrics.")
                    continue

                true_ranks = group['true_position']
                predicted_ranks = group['predicted_score_for_ranking'].rank(method='min')
                if len(true_ranks) > 1:
                    spearman_corr, _ = spearmanr(true_ranks, predicted_ranks)
                    kendall_corr, _ = kendalltau(true_ranks, predicted_ranks)
                    all_race_spearman_coeffs.append(spearman_corr)
                    all_race_kendall_coeffs.append(kendall_corr)
                    print(f"Race ({season_val}, {race_val}): Spearman = {spearman_corr:.4f}, Kendall = {kendall_corr:.4f}")
                else:
                    print(f"Race ({season_val}, {race_val}): Not enough data for rank correlation.")
            
            if all_race_spearman_coeffs: avg_spearman = np.nanmean(all_race_spearman_coeffs)
            if all_race_kendall_coeffs: avg_kendall = np.nanmean(all_race_kendall_coeffs)
            if avg_spearman is not None: print(f"\nAvg Spearman: {avg_spearman:.4f}")
            if avg_kendall is not None: print(f"Avg Kendall: {avg_kendall:.4f}")
        
        return {"test_rmse": test_rmse, "test_r2": test_r2, 
                "avg_spearman": avg_spearman, "avg_kendall": avg_kendall}

    def plot_training_progress(self, model_name_display=None):
        model_name_display = model_name_display or self.__class__.__name__
        if not self.evals_result or not isinstance(self.evals_result, dict):
            print("No evaluation results (or invalid format) to plot for training progress.")
            return None

        model_params_for_plot = self.model.get_params() if self.model else {}
        model_eval_metrics = model_params_for_plot.get('eval_metric', ["rmse"])
        if not isinstance(model_eval_metrics, list): model_eval_metrics = [model_eval_metrics]
        
        num_metrics = len(model_eval_metrics)
        if num_metrics == 0: 
            print("No eval_metric defined for plotting training progress.")
            return None

        fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics), squeeze=False)

        for i, metric_name in enumerate(model_eval_metrics):
            ax = axs[i, 0]
            if metric_name not in self.evals_result.get('validation_0', {}) or \
               metric_name not in self.evals_result.get('validation_1', {}):
                print(f"Metric '{metric_name}' not found in evaluation results. Skipping this subplot.")
                continue

            epochs = len(self.evals_result['validation_0'][metric_name])
            x_axis = range(0, epochs)
            
            ax.plot(x_axis, self.evals_result['validation_0'][metric_name], label=f'Train {metric_name.upper()}')
            ax.plot(x_axis, self.evals_result['validation_1'][metric_name], label=f'Test {metric_name.upper()}')
            ax.legend()
            ax.set_ylabel(metric_name.upper())
            ax.set_xlabel('Boosting Round')
            ax.set_title(f'{model_name_display} Training Progress ({metric_name.upper()})')
            ax.grid(True)
        
        plt.tight_layout()
        return fig

    def plot_last_race_comparison(self, model_name_display=None):
        """Plots predicted vs actual vs grid positions for the last race in the test set."""
        model_name_display = model_name_display or self.__class__.__name__
        if self.test_set_full_info_df is None or self.test_set_full_info_df.empty or \
           self.test_races_identifiers is None or self.test_races_identifiers.empty:
            print("No test set data available for last race comparison.")
            return None

        last_test_race_season = self.test_races_identifiers.iloc[-1]['season']
        last_test_race_round = self.test_races_identifiers.iloc[-1]['race']
        
        last_race_df = self.test_set_full_info_df[
            (self.test_set_full_info_df['season'] == last_test_race_season) &
            (self.test_set_full_info_df['race'] == last_test_race_round)
        ].copy() 

        if last_race_df.empty:
            print(f"No data found in test_set_full_info_df for S{last_test_race_season} R{last_test_race_round}.")
            return None
        
        # print(f"DEBUG_PLOT: Columns in last_race_df at start of plot_last_race_comparison: {last_race_df.columns.tolist()}")
        # if 'grid_position' in last_race_df.columns:
        #     print(f"DEBUG_PLOT: grid_position in last_race_df. Values: {last_race_df['grid_position'].tolist()}")
        #     print(f"DEBUG_PLOT: grid_position.notna().any(): {last_race_df['grid_position'].notna().any()}")
        # else:
        #     print("DEBUG_PLOT: grid_position NOT in last_race_df.columns at start of plot_last_race_comparison")


        if 'true_position' not in last_race_df.columns:
            if self.y_test is not None and last_race_df.index.isin(self.y_test.index).all():
                last_race_df['true_position'] = self.y_test.loc[last_race_df.index]
            else:
                original_indices = last_race_df.index
                if self.merged_df is not None and original_indices.isin(self.merged_df.index).all():
                    last_race_df['true_position'] = self.merged_df.loc[original_indices, self.target_column]
                else:
                    print("'true_position' missing and could not be aligned/reconstructed.")
                    return None


        if 'predicted_score_for_ranking' not in last_race_df.columns:
            if self.X_test is not None and last_race_df.index.isin(self.X_test.index).all():
                last_race_X_test_data = self.X_test.loc[last_race_df.index]
                if not last_race_X_test_data.empty:
                     last_race_df['predicted_score_for_ranking'] = self.predict(last_race_X_test_data)
                else:
                    print("Could not find matching X_test data for the last race plot to make predictions.")
                    return None
            else:
                print("Predictions not available (X_test issue or alignment) for last race plot.")
                return None
            
        last_race_df['predicted_rank'] = last_race_df['predicted_score_for_ranking'].rank(method='min')
        
        # --- MODIFICATION FOR SORTING ---
        if 'true_position' in last_race_df.columns and last_race_df['true_position'].notna().any():
            last_race_df_sorted = last_race_df.sort_values(by='true_position', na_position='last')
            print(f"DEBUG_PLOT: Sorting last race plot by 'true_position'.")
        elif 'grid_position' in last_race_df.columns and last_race_df['grid_position'].notna().any(): 
            print(f"DEBUG_PLOT: Sorting last race plot by 'grid_position' as 'true_position' was missing or all NaN.")
            last_race_df_sorted = last_race_df.sort_values(by='grid_position', na_position='last')
        else:
            print("Error: Neither 'true_position' nor 'grid_position' available for sorting last race plot, or both are all NaN.")
            return None
        # --- END MODIFICATION ---

        if 'true_position' not in last_race_df_sorted.columns: 
            print("'true_position' not found in data for last race plot after sorting.")
            return None
            
        fig, ax = plt.subplots(figsize=(14, 9))
        y_ticks_labels = [f"{name} (S:{s} R:{r} D:{did})" for name, s, r, did in zip(
            last_race_df_sorted.get('driver_name', last_race_df_sorted['driver_id']),
            last_race_df_sorted['season'],
            last_race_df_sorted['race'],
            last_race_df_sorted['driver_id']
        )]
        num_drivers = len(last_race_df_sorted)
        
        if 'grid_position' in last_race_df_sorted.columns and last_race_df_sorted['grid_position'].notna().any():
            ax.scatter(last_race_df_sorted['grid_position'], np.arange(num_drivers),
                       marker='s', color='green', label='Grid Position', s=100, zorder=2, alpha=0.7)
        # else:
            # print("DEBUG_PLOT: Not plotting grid_position scatter as it's missing or all NaN in last_race_df_sorted.")


        ax.scatter(last_race_df_sorted['true_position'], np.arange(num_drivers), 
                   marker='o', color='blue', label='Actual Position', s=100, zorder=3)
        
        ax.scatter(last_race_df_sorted['predicted_rank'], np.arange(num_drivers), 
                   marker='x', color='red', label='Predicted Rank', s=100, zorder=3)

        if 'grid_position' in last_race_df_sorted.columns and last_race_df_sorted['grid_position'].notna().any():
            for i in range(num_drivers):
                if pd.notna(last_race_df_sorted['grid_position'].iloc[i]) and pd.notna(last_race_df_sorted['true_position'].iloc[i]):
                    ax.plot([last_race_df_sorted['grid_position'].iloc[i], last_race_df_sorted['true_position'].iloc[i]],
                            [i, i], color='lightgreen', linestyle=':', linewidth=1.0, zorder=1)
        
        for i in range(num_drivers):
            if pd.notna(last_race_df_sorted['true_position'].iloc[i]) and pd.notna(last_race_df_sorted['predicted_rank'].iloc[i]):
                ax.plot([last_race_df_sorted['true_position'].iloc[i], last_race_df_sorted['predicted_rank'].iloc[i]],
                        [i, i], color='gray', linestyle='--', linewidth=0.8, zorder=1)

        ax.set_yticks(np.arange(num_drivers))
        ax.set_yticklabels(y_ticks_labels)
        ax.invert_yaxis() 
        ax.set_xlabel('Position / Rank')
        ax.set_ylabel('Driver (Season, Race, ID)')
        ax.set_title(f'{model_name_display}: Grid vs Actual vs Predicted Ranks for Last Test Race\n(S{last_test_race_season} R{last_test_race_round})')
        ax.legend(loc='best')
        plt.grid(True, axis='x', linestyle=':', alpha=0.7)
        
        max_val_for_xlim = 0
        if 'grid_position' in last_race_df_sorted and last_race_df_sorted['grid_position'].notna().any():
            max_val_for_xlim = max(max_val_for_xlim, last_race_df_sorted['grid_position'].max(skipna=True))
        if 'true_position' in last_race_df_sorted and last_race_df_sorted['true_position'].notna().any():
            max_val_for_xlim = max(max_val_for_xlim, last_race_df_sorted['true_position'].max(skipna=True))
        if 'predicted_rank' in last_race_df_sorted and last_race_df_sorted['predicted_rank'].notna().any():
            max_val_for_xlim = max(max_val_for_xlim, last_race_df_sorted['predicted_rank'].max(skipna=True))
        
        if max_val_for_xlim > 0 :
            ax.set_xticks(np.arange(1, int(max_val_for_xlim) + 2)) 
            ax.set_xlim(0.5, int(max_val_for_xlim) + 1.5)
        else: 
            ax.set_xticks(np.arange(1, 21)) 
            ax.set_xlim(0.5, 20.5)

        plt.tight_layout()
        return fig
    
    def run_pipeline(self):
        """Runs the full pipeline: load, prepare, train, evaluate, and plot."""
        self.load_and_prepare_data()
        self.train_model()
        metrics = self.evaluate_model()
        
        fig_progress = self.plot_training_progress()
        if fig_progress: plt.show()
        
        fig_last_race = self.plot_last_race_comparison()
        if fig_last_race: plt.show()
            
        importances = self.get_feature_importances()
        if importances is not None:
            print("\n--- Feature Importances (Top 10) ---")
            print(importances.head(10))
        
        print("\n--- Model building pipeline finished ---")
        return metrics
