import logging
import pandas as pd
import numpy as np

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


logger = logging.getLogger(__name__)


def mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between the true and predicted values.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between the true and predicted values.
    """
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def test_correlation_exog_vars(history_data, exog_vars):
    """
    Test for correlation between the target variable 'y' and a list of exogenous variables.
    """
    
    logger.info(f"Testing for correlation between 'y' in {exog_vars}: ")
    correlations = {}

    # Calculate the correlation of each exogenous variable with the target
    for exog in exog_vars:
        corr, _ = pearsonr(history_data[exog], history_data['y'])
        correlations[exog] = corr

    # Print out the correlation coefficients
    for exog, corr in correlations.items():
        logger.info(f"Correlation between 'y' and {exog}: {corr}")

    return correlations


def test_multicollinearity(history_data, exog_vars):
    """
    Test for multicollinearity between a list of exogenous variables.
    """
    
    logger.info(f"Testing for multicollinearity between {exog_vars}: ")
    
    X = history_data[exog_vars]
    X = add_constant(X)  # add a constant column for the intercept

    # Calculate Variance Inflation Factor (VIF) for each exogenous variable
    vif = pd.DataFrame()
    vif['variable'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    logger.info(vif)

    return vif


def create_performance_table(experiments):
    """
    Create a performance table for a list of experiments.
    """
    
    all_rows = []

    for experiment in experiments:
        model_names = [type(model).__name__ for model in experiment.models]

        for model_name in model_names:
            if model_name in experiment.losses.index:
                loss = experiment.losses.at[model_name, 'mean_absolute_error']

                # Create a dictionary for each row and add it to the list
                row = {
                    'Experiment': experiment.experiment_name,
                    'Model': model_name,
                    'Mean Absolute Error': loss,
                    'Train Time (s)': experiment.train_cpu_time,  # Assuming total time for all models
                    'Predict Time (s)': experiment.predict_cpu_time  # Assuming total time for all models
                }
                all_rows.append(row)

    all_data = pd.DataFrame(all_rows)

    logger.info("=========Final Performance=========")
    logger.info(all_data)
    return all_data

    
def plot_error_vs_time(performance_table):
    """
    Plot the Mean Absolute Error vs Train Time and Mean Absolute Error vs Predict Time.
    """
    # Plotting Error vs Train Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=performance_table, x='Train Time (s)', y='Mean Absolute Error', hue='Model', style='Experiment', s=100)
    plt.title('Mean Absolute Error vs Train Time')
    plt.xlabel('Train Time (seconds)')
    plt.ylabel('Mean Absolute Error')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    # Plotting Error vs Predict Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=performance_table, x='Predict Time (s)', y='Mean Absolute Error', hue='Model', style='Experiment', s=100)
    plt.title('Mean Absolute Error vs Predict Time')
    plt.xlabel('Predict Time (seconds)')
    plt.ylabel('Mean Absolute Error')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()