import os
from omegaconf import DictConfig, ListConfig
import mlflow
import hydra
import logging

# 実験ログを作成するクラス
class ExperimentRecorder():
    # 初期設定
    def __init__(self, experiment_name,run_name=None):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        logging.basicConfig(level=logging.WARN)
    
    # カレントディレクトリを得る
    def get_things(self):
        org_dir = hydra.utils.get_original_cwd()
        run_dir = os.path.abspath('.')
        return org_dir, run_dir, logging.getLogger(__name__)

    #クラス内変数を作る？下記の関数を使用してパラメータをいれる箱？
    def log_all_params(self, root_param):
        self._explore_recursive('', root_param)

    # パラメータの名前と値をconf.ymlから探していれる関数
    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}{k}.', v)
                else:
                    mlflow.log_param(f'{parent_name}{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                mlflow.log_param(f'{parent_name}{i}', v)
        else:
            print('ignored to log param:', element)
    
    def end_run(self):
        mlflow.end_run()