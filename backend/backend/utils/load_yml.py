from envyaml import EnvYAML


def load_config(yml_path) -> dict:
    """Loads a .yml file with environment variables evaluated."""
    env = EnvYAML(yml_path)
    return env.export()
