import os
import sys
from omegaconf import DictConfig, ListConfig
import logging

import hydra
from hydra import utils
import mlflow

def _explore_recursive(parent_name, element):
        if isinstance(element, DictConfig):
            print("element:",element)
            for k, v in element.items():
                print("k:",k,"v:",v)

                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    print(f'{parent_name}{k}.', v)
                else:
                    print(f'{parent_name}{k}', v)
        elif isinstance(element, ListConfig):
            print(element)
            for i, v in enumerate(element):
                print("i:",i,"v:",v)
                print(f'{parent_name}{i}', v)
        else:
            print('ignored to log param:', element)

@hydra.main(config_path='config.yaml')
def main(cfg):
    dict = {'model':{'node1': 128, 'node2': 64}}

    _explore_recursive('model.',dict)

if __name__ == '__main__':
    main()