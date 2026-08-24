"""
Script d'entrée : orchestre la construction des datasets, l'entraînement,
l'évaluation et la visualisation du modèle d'occupation.

@author: Myriam Lopez
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.pytorch

import datasets
from datasets import build_train_test_datasets
from training import train_model
from evaluation import evaluate_model, plot_example

# à basculer selon l'environnement d'exécution : False = local, True = SageMaker/S3
SAGE_ENV = True
datasets.SAGE_ENV = SAGE_ENV

mlflow.set_tracking_uri("arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-CZPP4GNRF7VT")
mlflow.set_experiment("Occupation-model-training")

def _table_to_metrics(df, key_cols, value_cols, prefix):
    """Aplati un DataFrame agrégé (peu de lignes) en métriques MLflow scalaires : prefix/clé.../colonne."""
    metrics = {}
    for _, row in df.iterrows():
        key = "/".join(str(row[c]) for c in key_cols)
        for col in value_cols:
            if pd.notna(row[col]):
                metrics[f"{prefix}/{key}/{col}"] = float(row[col])
    return metrics


def log_evaluation_run(model_type, model, artifacts, results, run_name, note=None, dataset_tags=None):
    with mlflow.start_run(run_name=run_name):
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        if dataset_tags:
            for k, v in dataset_tags.items():
                mlflow.set_tag(k, v)
        mlflow.log_param("model_type", model_type)
        if model_type == 'lightgbm':
            mlflow.log_params(model.get_params())
        else:
            mlflow.log_params({'seq_len': artifacts['seq_len'], 'stride': artifacts['stride']})

        metrics = dict(results['metrics'])
        # tables agrégées (peu de lignes) : loggées comme vraies métriques MLflow, exploitables dans l'UI
        metrics.update(_table_to_metrics(results['bias_by_hour'], ['hour'], ['mbe', 'nmae'], 'bias_by_hour'))
        metrics.update(_table_to_metrics(results['bias_by_strata'], ['jour', 'strate'], ['mbe', 'nmae'], 'bias_by_strata'))
        metrics.update(_table_to_metrics(results['fwhm_by_weekday'], ['jour'], ['fwhm_true', 'fwhm_pred', 'diff'], 'fwhm_by_weekday'))
        mlflow.log_metrics(metrics)

        # fwhm_by_day : une ligne par (site, jour), trop granulaire pour des métriques scalaires -> reste en CSV
        for table_name in ('bias_by_hour', 'bias_by_strata', 'fwhm_by_day', 'fwhm_by_weekday'):
            mlflow.log_text(results[table_name].to_csv(index=False), f"{table_name}.csv")

        if model_type == 'lightgbm':
            mlflow.lightgbm.log_model(model, name="model")
        else:
            mlflow.pytorch.log_model(model, name="model")

        fig = plot_example(model_type, model, artifacts)
        mlflow.log_figure(fig, "prediction_example.png")
        plt.close(fig)


def main():
    preprocess = 'rolling_minmax'
    train_df, test_df = build_train_test_datasets(preprocess=preprocess)
    model_type = 'lightgbm'  # ou 'lightgbm'
    model, artifacts = train_model(model_type, train_df, test_df)

    # evaluate_model affiche déjà les scores globaux (dtw, rmse_weekly, ...) et les tables agrégées
    # (bias_by_strata, fwhm_by_weekday) ; le detail par heure/jour reste dispo pour inspection/plot
    results = evaluate_model(model_type, model, artifacts)
    print(results['bias_by_hour'])
    print(results['fwhm_by_day'])

    dataset_tags = {
        "dataset.n_sites_test": artifacts['test_df']['site_id'].nunique(),
        "dataset.preprocess": preprocess,
    }
    log_evaluation_run(
        model_type, model, artifacts, results,
        run_name=f"{model_type}_{datetime.now():%Y%m%d_%H%M%S}",
        dataset_tags=dataset_tags,
    )

if __name__ == '__main__':
    main()
