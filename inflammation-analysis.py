#!/usr/bin/env python3
"""Software for managing and analysing patients' inflammation data in our imaginary hospital."""

import argparse
import os

from inflammation import models, views
from inflammation.compute_data import analyse_data, CSVDataSource, JSONDataSource


def main(args):
    """The MVC Controller of the patient inflammation data system.

    The Controller is responsible for:
    - selecting the necessary models and views for the current task
    - passing data between models and views
    """
    infiles = args.infiles
    if not isinstance(infiles, list):
        infiles = [args.infiles]

    if args.full_data_analysis:
        _, extension = os.path.splitext(infiles[0])

        if extension == '.json':
            data_input = JSONDataSource(os.path.dirname(infiles[0]))
        elif extension == '.csv':
            data_input = CSVDataSource(os.path.dirname(infiles[0]))
        else:
            raise ValueError(f'Unsupported data file format: {extension}')
        analyse_data(data_input)

        graph_data = {
            'standard deviation by day': data_input,
        }
        views.visualize(graph_data)


    for filename in infiles:
        inflammation_data = models.load_csv(filename)

        view_data = {
            'average': models.daily_mean(inflammation_data),
            'max': models.daily_max(inflammation_data),
            'min': models.daily_min(inflammation_data)
        }

        views.visualize(view_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A basic patient inflammation pythondata management system')

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Input CSV(s) containing inflammation series for each patient')

    parser.add_argument(
        '--full-data-analysis',
        action='store_true',
        dest='full_data_analysis')

    args = parser.parse_args()

    main(args)
