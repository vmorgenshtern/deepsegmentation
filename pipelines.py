import numpy as np
import os
import os.path
import random

# User imports
from PolyhedronGenerator import polyhedronGenerator
from structureHelpers import write_config
from generator_config import generator_config
from utils.mode import Mode

DEFAULT_SEED = 42
DEFAULT_CONFIG_FILENAME = 'config.json'


def init_rng(seed):
    """
    To make results reproducible fix seed of RNG
    """
    random.seed(seed)


def simulation_pipeline(params, batch_size, dataset_name, mode, seed):
    """
    Main loop of the generator. Generates point clouds and saves them
    to the input data folder.

    Input:
    params       -- global parameters like paths, filenames etc.
    batch_size   -- number of point clouds to generate
    dataset_name -- name of the dataset to be produced
    mode         -- Mode.TRAIN, Mode.TEST or Mode.PREDICT
    seed         -- seed for the RNG (optional)
    """
    init_rng(seed or DEFAULT_SEED)

    dataset_path = setup_data_dir(params, dataset_name)
    gen_cfg = generator_config()

    # Serialize config to dataset folder
    write_config({
        'global': params,
        'simulator': gen_cfg,
        'others': {
            'seed': seed
        }
    }, os.path.join(dataset_path, DEFAULT_CONFIG_FILENAME))

    # Generator main loop
    for i in range(batch_size):
        'get dictionary of polyhedra vertices with randomized vertex scaling'
        polyhedra_dict = get_polyhedra_dict(gen_cfg, mode=mode)

        'get randomized settings'
        rnd_key = random.choice([name for name in polyhedra_dict.keys()])
        rnd_nr_points = int(np.random.uniform(gen_cfg['nr_points']['lower_bound'],
                                              gen_cfg['nr_points']['upper_bound']))
        rnd_scaling = np.random.uniform(gen_cfg['global_scaling']['lower_bound'],
                                        gen_cfg['global_scaling']['upper_bound'])

        if np.random.uniform(0, 1) <= gen_cfg['edge_rounding']['apply_probability']:
            k_round_edges = int(rnd_nr_points / np.random.uniform(gen_cfg['edge_rounding']['fraction_lower_bound'],
                                                                  gen_cfg['edge_rounding']['fraction_upper_bound']))
        else:
            k_round_edges = 0

        if np.random.uniform(0, 1) <= gen_cfg['roll_pitch_yaw_deg']['apply_probability']:
            roll = np.random.uniform(gen_cfg['roll_pitch_yaw_deg']['lower_bound'],
                                     gen_cfg['roll_pitch_yaw_deg']['upper_bound'])
            pitch = np.random.uniform(gen_cfg['roll_pitch_yaw_deg']['lower_bound'],
                                     gen_cfg['roll_pitch_yaw_deg']['upper_bound'])
            yaw = np.random.uniform(gen_cfg['roll_pitch_yaw_deg']['lower_bound'],
                                     gen_cfg['roll_pitch_yaw_deg']['upper_bound'])
            roll_pitch_yaw = [roll, pitch, yaw]

        else:
            roll_pitch_yaw = [0, 0, 0]

        squash_x = np.random.uniform(gen_cfg['squash_xy']['squash_x_lower_bound'],
                                     gen_cfg['squash_xy']['squash_x_upper_bound'])

        squash_y = np.random.uniform(gen_cfg['squash_xy']['squash_y_lower_bound'],
                                     gen_cfg['squash_xy']['squash_y_upper_bound'])

        poly = polyhedronGenerator.polyhedronGenerator(polyhedra_dict[rnd_key], nr_points_polyhedron=rnd_nr_points,
                                        path=dataset_path, name=rnd_key, mode=gen_cfg['mode'],
                                        variance=gen_cfg['noise_params']['variance'], scaling=rnd_scaling,
                                        roll_pitch_yaw=roll_pitch_yaw, k_round_edges=k_round_edges,
                                        squash_xy=[squash_x, squash_y])

        points = poly.convexHullGenerate()

        'save npy file and store a hmtl file in the same folder for later observation'
        poly.visualize(jupyter=False, auto_open=False)
        poly.save()

def setup_data_dir(params, dataset_name):
    dataset_path = os.path.join(os.path.abspath(params['input_data_path']), dataset_name)

    try:
        os.makedirs(dataset_path)
    except OSError as e:
        raise Exception(("Directory %s already exists. Please choose an unused dataset " +
                "identifier") % (dataset_path,))

    return dataset_path


def get_polyhedra_dict(generator_cfg, mode):
    """
    This function returns a dictionary of polyhedra and corresponding vertices.
    Scaling of vertices is randomized by configuration file.

    Inputs:
        generator_cfg(dict): Dictionary with generator configuration
        mode: Mode.TRAIN, Mode.TEST
    Outputs:
        dict_polyhedra(dict): Dictionary of polyhedra vertices
    """
    'get vertex scales'
    vertex_scales = []
    vertex_scales_keys = generator_cfg['vertex_scales'].keys()
    for vertex_scale in vertex_scales_keys:
        vertex_scales.append(np.random.uniform(generator_cfg['vertex_scales'][vertex_scale]['lower_bound'],
                                               generator_cfg['vertex_scales'][vertex_scale]['upper_bound']))
    'get dictionary of available polyhedra'
    dict_polyhedra = polyhedronGenerator.return_polyhedra(*vertex_scales, mode=mode)

    return dict_polyhedra