"""
Utility functions used by more than one module
"""
from matplotlib import pyplot as plt


def plot_text(coord, str_label, color=(1., 1., 1.)):
    """
    Wrap plt.text to get a consistent look
    """
    plt.text(coord[0].item(),
             coord[1].item(),
             str_label,
             ha="center",
             va="center",
             fontsize='xx-small',
             bbox=dict(boxstyle="round", fc=color, ec=None))
