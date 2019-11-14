import sys
sys.path.append('../')

import json
import os
import shutil

def read_config(cfg_path):
    """
    Read a json config file

    Inputs:     cfg_path(string): Path to json config file
    """
    with open(cfg_path, 'r') as f:
        params = json.load(f)

    return params


def write_config(params, cfg_path, sort_keys=False):
    """
    Write a json config file

    Inputs: params(Dictionary): Model parameters
            cfg_path(string): Path to config file
            sort_keys(bool): Sort json output by key or not
    """
    with open(cfg_path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=sort_keys)


def create_experiment(experiment_name):
    """
    This method sets the structure for a new experiment. This allows to store experiments and related data separately,
    so different experiments can be compared.

    Inputs:
        experiment_name(string): Name of the experiment
    Outputs:
        params(dict): Dictionary containing the configuration of the current experiment

    """
    params = read_config('./config.json')
    params['experiment_name'] = experiment_name
    create_experiment_folders(params)
    cfg_file_name = params['experiment_name'] + '_config.json'
    cfg_path = os.path.join(params['network_output_path'], cfg_file_name)
    params['cfg_path'] = cfg_path
    write_config(params, cfg_path)

    return params


def create_experiment_folders(params):
    """
    Create new experiment based on given configuration parameters if directories do not exist yet.
    """
    try:
        path_keynames = ["network_output_path", "output_data_path", "tf_logs_path"]
        for key in path_keynames:
            params[key] = os.path.join(params[key], params['experiment_name'])
            os.makedirs(params[key])

    except:
        raise Exception("Experiment already exist. Please try a different experiment name")


def open_experiment(experiment_name):
    '''Open Existing Experiments'''

    default_params = read_config('./config.json')
    cfg_file_name = experiment_name + '_config.json'
    cfg_path = os.path.join(default_params['network_output_path'], experiment_name, cfg_file_name)
    params = read_config(cfg_path)

    return params


def delete_experiment(experiment_name):
    '''Delete Existing Experiment folder'''

    default_params = read_config('./config.json')
    cfg_file_name = experiment_name + '_config.json'
    cfg_path = os.path.join(default_params['network_output_path'], experiment_name, cfg_file_name)

    params = read_config(cfg_path)

    path_keynames = ["network_output_path", "output_data_path", "tf_logs_path"]
    for key in path_keynames:
        shutil.rmtree(params[key])


def create_experiment_folders(params):
    try:
        path_keynames = ["network_output_path", "output_data_path", "tf_logs_path"]
        for key in path_keynames:
            params[key] = os.path.join(params[key], params['experiment_name'])
            os.makedirs(params[key])
    except:
        raise Exception("Experiment already exist. Please try a different experiment name")


def open_experiment(experiment_name):
    '''Open Existing Experiments'''

    default_params = read_config('./config.json')
    cfg_file_name = experiment_name + '_config.json'
    cfg_path = os.path.join(default_params['network_output_path'], experiment_name, cfg_file_name)
    params = read_config(cfg_path)

    return params


def delete_experiment(experiment_name):
    '''Delete Existing Experiment folder'''

    default_params = read_config('./config.json')
    cfg_file_name = experiment_name + '_config.json'
    cfg_path = os.path.join(default_params['network_output_path'], experiment_name, cfg_file_name)

    params = read_config(cfg_path)

    path_keynames = ["network_output_path", "output_data_path", "tf_logs_path"]
    for key in path_keynames:
        shutil.rmtree(params[key])

def create_retrain_experiment(experiment_name, source_pth_file_path):
    params=create_experiment(experiment_name)
    params['Network']['retrain'] = True

    destination_pth_file_path=os.path.join(params['network_output_path'],'pretrained_model.pth')
    params['Network']['pretrain_model_path'] = destination_pth_file_path
    shutil.copy(source_pth_file_path,destination_pth_file_path)

    write_config(params, params['cfg_path'])
    return params
