"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
import json


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    : returns: Numpy array of filename
    """
    return np.loadtxt(fname=filename, delimiter=',')

def load_json(filename):
    """Load a numpy array from a JSON document.

    Expected format:
    [
        {
            observations: [0, 1]
        },
        {
            observations: [0, 2]
        }
    ]

    :param filename: Filename of CSV to load

    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: 2D array of inflammation data
    :return: daily mean of inflammation data
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: 2D array of inflammation data
    :return: daily max of inflammation data
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array."""
    return np.min(data, axis=0)


def daily_std(data):
    """Computes and returns standard deviation for data."""
    mmm = np.mean(data, axis=0)
    devs = []
    for entry in data:
        devs.append((entry - mmm) * (entry - mmm))

    std_dev = sum(devs) / len(data)
    return std_dev

def patient_normalise(data):
    """Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0."""
    if np.any(data < 0):
        raise ValueError('Negative values are not allowed')
    if not isinstance(data, np.ndarray):
        raise TypeError('Data must be an Numpy array')
    if len(data.shape) != 2:
        raise ValueError('Shape of inflammation array should be 2-dimensional')
    max_data = np.nanmax(data, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised

