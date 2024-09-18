config = {
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 1000,
        'save_every': 1,
        'regularization_lambda': 0.1,
    },
    'model': {
        'num_atomic_numbers': 100,
        'num_gen_steps': 1000,
        'gaussian_stop': 20.0,
        'basis_width_scalar': 3.0,
        'radius_rate': 3.0,
        'step_size_start': 0.5,
        'step_size_end': 0.0005,
        'temp_start': 1.0,
        'temp_end': 0.001,
        'expected_density': 0.05,
        'energy_penalty': 1.0,
    },
}
