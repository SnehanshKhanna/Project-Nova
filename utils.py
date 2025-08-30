import numpy as np

def map_to_nova_score(prob):
    """Map probability (0-1) to Nova Score (300-850)."""
    return int(300 + prob * 550)
