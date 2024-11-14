"""Utility functions for the Information Extraction demo."""

import importlib

RESOURCE_PATH = importlib.resources.files("clinlp_apps.ie_demo.resources")


def simple_label(label: str) -> str:
    """
    Create a simple label.

    Parameters
    ----------
    label
        The input label.

    Returns
    -------
        The simple label.
    """
    return label.lower().replace(" ", "_")
