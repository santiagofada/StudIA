import yaml
from pathlib import Path

def load_config(fname="config.yaml"):
    with open(Path(__file__).parent / fname, encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()  # objeto global accesible desde cualquier módulo
