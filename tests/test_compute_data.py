from unittest.mock import Mock

from pathlib import Path
import numpy as np
import numpy.testing as npt

from inflammation.compute_data import CSVDataSource


# def test_compute_data_mock_source():
#     from inflammation.compute_data import analyse_data
#     data_input = Mock()
#     mock_array = [[[0, 1, 3, 6, 2]],
#                   [[2, 1, 4, 2, 9]]]
#
#     data_input.load_inflammation_data.return_value = mock_array
#
#     result = analyse_data(data_input)
#     npt.assert_array_almost_equal(result, [1, 0, 0.5, 2, 3.5])

def test_analyse_data():
    from inflammation.compute_data import analyse_data
    path = Path.cwd() / "../data"
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    expected_result = [0.,         0.22510286, 0.18157299, 0.1264423,  0.9495481,  0.27118211,
                       0.25104719, 0.22330897, 0.89680503, 0.21573875, 1.24235548, 0.63042094,
                       1.57511696, 2.18850242, 0.3729574,  0.69395538, 2.52365162, 0.3179312,
                       1.22850657, 1.63149639, 2.45861227, 1.55556052, 2.8214853,  0.92117578,
                       0.76176979, 2.18346188, 0.55368435, 1.78441632, 0.26549221, 1.43938417,
                       0.78959769, 0.64913879, 1.16078544, 0.42417995, 0.36019114, 0.80801707,
                       0.50323031, 0.47574665, 0.45197398, 0.22070227]

    npt.assert_array_almost_equal(result, expected_result)



def test_compute_standard_deviation_by_day():
    from inflammation.compute_data import compute_standard_deviation_by_day

