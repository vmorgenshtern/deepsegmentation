def generator_config():
    """
    Function returns a dictionary that is used to randomize the polyhedron generation.
    """

    return{
        'mode': 'clean', #possible entries: clean, noisy
        'noise_params': {
            'variance': 0.0001,
            'mean': 0,  #no effect currently
        },
        'nr_points': {
            'upper_bound': 50000, #framework works also with 100,000 points, but due to different colour clustered point clouds may become too big to display
            'lower_bound': 5000
        },
        'global_scaling': {
            'upper_bound': 1,
            'lower_bound': 1
        },
        'vertex_scales': {
            'C0': {
                'upper_bound': 1,
                'lower_bound': 1
                },
            'C1': {
                'upper_bound': 1,
                'lower_bound': 1
                },
            'C2': {
                'upper_bound': 1,
                'lower_bound': 1
                },
            'C3': {
                'upper_bound': 1,
                'lower_bound': 1
                },
            'C4': {
                'upper_bound': 1,
                'lower_bound': 1
                },
            'C5': {
                'upper_bound': 1,
                'lower_bound': 1
                },
        },
        'include_spheres': False,
        'edge_rounding': {
            'apply_probability': 0.5,
            'fraction_upper_bound': 10000, #used as k_rounding = nr_points/fraction
            'fraction_lower_bound': 100,
        },
        'roll_pitch_yaw_deg': {
            'apply_probability': 1,
            'upper_bound': 360,
            'lower_bound': 0
        },
        'squash_xy': {
           'squash_x_upper_bound': 1,
           'squash_x_lower_bound': 0.5,
           'squash_y_upper_bound': 1,
           'squash_y_lower_bound': 0.5
        }
    }
