import pickle
import collections
import numpy as np


def apply_mask(data, mask):
    """ Apply the given mask to all arrays in the dataset dictionary 'data'

    Parameters
    ----------
    data : dict
        dictionary of data arrays
    mask : array_like
        data is kept at locations where the mask is True

    Returns
    -------
    data_new : dict
        dictionary of data array with mask applied.

    """
    data_new = {}
    for k, v in data.items():
        data_new[k] = v[mask]
    return data_new


def get_data_sorted_by_duration(data_dir):
    """Load all mangotether data files (all .pkl files) in the given directory and
    return an OrderedDict where the items are ordered by duration.

    Parameters
    ----------
    data_dir : Path
        the directory containing the data files. 

    """
    duration_and_data = []
    for f in data_dir.iterdir():
        if f.suffix == '.pkl':
            data = load_data(f)
            duration = get_duration(data)
            duration_and_data.append((duration, data))
    duration_and_data.sort()
    data_dict = collections.OrderedDict(duration_and_data)
    return data_dict


def load_data(filename):
    """ Load pickled dataset from the given file

    Parameters
    ----------
    filename : str
        the name of the file to load data from

    Returns
    -------
    data : dataset loaded from file.

    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def fix_setpt_error(data):
    """ Fixes setput error in magnotether dataset. Sometimes setpt is < 0
    where is it should always be > 0. 

    Parameters
    ----------
    data : dict
        dictionary of data arrays

    Retruns
    -------
    data : dict
        dictionary of data arrays with setpt issue fixed.

    """
    mask = data['setpt'] < 0
    data['setpt'][mask] = 0
    return data

def get_duration(data):
    """ Get stimulus duration from magnotether dataset.

    Parameters
    ----------
    data : dict
        dictionary of data arrays

    Returns
    -------
    duration : float
        stimulus duration for trial
    """
    mask = np.logical_and(data['setpt'] > 0, np.logical_not(data['disable']))
    t = data['t'][mask]
    duration = int(np.round(t[-1] - t[0]))
    return duration

def fix_setpt_error(data):
    """ Fixes setput error in magnotether dataset. Sometimes setpt is < 0
    where is it should always be > 0. 

    Parameters
    ----------
    data : dict
        dictionary of data arrays

    Retruns
    -------
    data : dict
        dictionary of data arrays with setpt issue fixed.

    """
    mask = data['setpt'] < 0
    data['setpt'][mask] = 0
    return data


def resample_data(t, x, num_pts=10):
    """ Resample time series data so that data is uniformly spaced respect to x

    Parameters
    ----------
    t : ndarray
        1D array of time values  
    x : ndarray
        1D array of data values 
    num_pts : int (optional)
        number of points in resampled data
    epsilon : float (optional)
        window size parameter finding resampled t values must be in [0,1]

    Returns
    -------
    t_rs : ndarray
        1D array of resampled time values
    x_rs : ndarray
        1D array of resampled data values

    """
    x_rs = np.linspace(x.min(), x.max(), num_pts+2)[1:-1]
    h = x_rs[1] - x_rs[0]
    t_rs = np.zeros_like(x_rs)
    for i, value in enumerate(x_rs):
        ind = np.argmin(np.absolute(x - value))
        t_rs[i] = t[ind]
        #mask = np.logical_and(x > value-h, x < value+h) 
        #t_rs[i] = np.median(t[mask])

    return t_rs, x_rs






