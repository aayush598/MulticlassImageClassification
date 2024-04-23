import yaml
import itertools
from training import training  

def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters loaded from the YAML file.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_combinations(config):
    """
    Generate all possible combinations of parameters from the configuration.

    Args:
        config (dict): Configuration parameters.

    Returns:
        list: List of tuples containing all possible combinations of parameters.
    """
    return list(itertools.product(*config.values()))

if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config('config.yaml')

    # Generate combinations of parameters
    combinations = generate_combinations(config)

    # Iterate over combinations and call the training function
    for combination in combinations:
        training(combination)
        