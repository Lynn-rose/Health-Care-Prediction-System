import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score


class df_helper:
    def categorical_columns(self, df):
        """Returns a list of categorical columns in the dataframe."""
        return df.select_dtypes(include=['object', 'category']).columns
    
    def numerical_columns(self, df):
        """Returns a list of numerical columns in the dataframe."""
        return df.select_dtypes(include=[np.number]).columns
    def column_decimal_rounding(self, df, column, decimals):
        df[column] = round(df[column], decimals)
        return df
    
    def column_subset(self, df, columns):
        return df[columns]
    
    def convert_inches_to_meters(self, df, column, decimals):
        df[column] = df[column].astype(float)/39.3700787
        return self.column_decimal_rounding(df, column, decimals)

    def convert_lbs_to_kg(self, df, column, decimals):
        df[column] = df[column].astype(float)*0.453592
        return self.column_decimal_rounding(df, column, decimals)
    
    def compare_predictions(self, y_test, y_pred, n_values=10):
        for i in range(n_values):
            print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
    
    def drop_columns(self, df, columns):
        df.drop(columns=columns, axis=1, inplace=True)
        return df
    def drop_columns_copy(self, df, columns):
        return df.drop(columns=columns, axis=1)
    
    def encode(self, df):
        return pd.get_dummies(df, columns=self.categorical_columns(df))
    
    def evaluate_model(self, y_test, y_pred):
        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)

        # Print metrics
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")
        print(f"Median Absolute Error: {median_ae:.2f}")
        print(f"Explained Variance Score: {explained_variance:.2f}")

    def reduce_unique_values(self, df, column, n_values):
        counts = df[column].value_counts()
        top = counts.head(n_values).index.tolist()
        df[column] = df[column].apply(lambda x: x if x in top else 'Other')
        
    def rename_column(self, df, old_col, new_col):
        df.rename(columns={old_col: new_col}, inplace=True)
        return df
    
    def rename_columns(self, df, columns_dict):
        df.rename(columns=columns_dict, inplace=True)
        return df
    
    def xy_split(self, df, target, test_size=0.2, random_state=42):
        X = df.drop(target, axis=1)
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    #####################################################################################
    # VISUALIZATION FUNCTIONS
    #####################################################################################
    def boxplot(self, df, column):
        df[column].plot.box()
        plt.show();
    def qqplot(self, df, column):
        stats.probplot(df[column], dist="norm", plot=plt)
        plt.show();