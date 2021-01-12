import os
import warnings
import sys
import shutil

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from experiment_recorder import *
#from utils.experiment_recorder import ExperimentRecorder
import mlflow.sklearn


warnings.filterwarnings("ignore")
np.random.seed(40)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(cwd, logger, in_alpha, in_l1_ratio):

    # データ取得
    csv_url =\
        'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)

    # データ加工
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # パラメータ
    if float(in_alpha) is None:
        alpha = 0.5
    else:
        alpha = float(in_alpha)

    if float(in_l1_ratio) is None:
        l1_ratio = 0.5
    else:
        l1_ratio = float(in_l1_ratio)

    # 実行
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    # 精度
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out metrics
    logger.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    logger.info("  RMSE: %s" % rmse)
    logger.info("  MAE: %s" % mae)
    logger.info("  R2: %s" % r2)

    return lr, rmse, mae, r2


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig) -> None:
   
    # 実験開始し,ログ作成するクラス
    recorder = ExperimentRecorder('test5',run_name=f'alpha={cfg.alpha},l1={cfg.l1_ratio}')

    # カレントディレクトリをlogとして出力
    org_dir, run_dir, logger = recorder.get_things()  
    logger.info(f'cwd was {org_dir}...')              
    logger.info(f'running cwd is {run_dir}...')        

    # 学習
    lr, rmse, mae, r2 = train(org_dir, logger, cfg.alpha, cfg.l1_ratio)

    # パラメータをmlflowに保存 -> db
    recorder.log_all_params(cfg)   
    # 精度をmlflowに保存 -> db
    mlflow.log_metric("rmse", rmse) 
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    # モデルを保存 -> art
    mlflow.sklearn.log_model(lr, "model")
    # Hydra　-> art
    mlflow.log_artifact('.hydra/config.yaml',artifact_path='hydra')
    mlflow.log_artifact('.hydra/hydra.yaml',artifact_path='hydra')
    mlflow.log_artifact('.hydra/overrides.yaml',artifact_path='hydra')

    #output
    features = "rooms, zipcode, median_price, school_rating, transport"
    with open("features.txt", 'w') as f:
        f.write(features)

    mlflow.log_artifact("features.txt", artifact_path="features")

    # 4. Let's finish.
    recorder.end_run()
    
    # 5. remove outputs
    os.chdir("/Workdir")
    try:
        shutil.rmtree("multiruns")
    except:
        pass
    
    try:
        shutil.rmtree("outputs")
    except:
        pass
    

if __name__ == "__main__":
    main()
