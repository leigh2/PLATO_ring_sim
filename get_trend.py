#!/usr/bin/env python3

import numpy as np
from cffi import FFI

ffi = FFI()
ffi.cdef("""
void c_get_trend(const double *f,
                 const bool *ol,
                 const int arr_size,
                 const int kernel_size,
                 double *f_trend,
                 double *f_error);
""")
lib = ffi.dlopen("./libget_trend.so")


def get_trend(t, f, kernel_size, sigma_clip=False, data_mask=None):
    """ A simple attempt to remove trends in light curve data.

    Parameters:
    t : ndarray
        Observations times.
    f : ndarray
        Observed fluxes.
    kernel_size : int
        Number of points on either side of each time point inside which the
        systematics should be considered constant.
    sigma_clip : boolean; optional
        If True sigma clip the input data to remove outlying noise (signals),
        otherwise leave the data as-is. (Default False.)
    data_mask : boolean array; optional
        An additional masking of input time and flux arrays. Where mask values
        are False this data point will not be used for trend measurement.
        (Default None.)

    Returns:
    trend : ndarray
        Cleaned fluxes.
    error : ndarray
        Approximated error on cleaned fluxes.
    """

    # check input arrays
    t, f = map(lambda _a: np.asarray(_a), [t, f])
    if t.ndim!=1 or f.ndim!=1:
        raise RuntimeError("Time and flux arrays must be 1 dimensional")

    # sort the data chronologically
    order = np.argsort(t.copy())
    f = f[order]
    t = t[order]
    # store original order too so we can restore later
    orig_order = np.arange(t.size)[order]

    # deal with data mask
    if data_mask is not None:
        data_mask = np.asarray(data_mask)
        if data_mask.ndim!=1:
            raise RuntimeError("Data mask must be 1 dimensional if provided")
        outlier = data_mask[order] # order chronologically
    else:
        outlier = np.zeros(t.size, dtype=np.bool)

    # remove clear outliers (e.g. transit points)
    if sigma_clip:
        for level in [10,9,8,7,6,5]: # reduce cut level each time
            stdev = np.std(f[~outlier])
            outlier[f > level*stdev] = True

    # initialise trend arrays
    f_trend = f.copy()
    f_error = np.zeros(f.size, dtype=np.float64)

    # make arrays contiguous
    _contig = lambda arr: np.ascontiguousarray(arr)
    f, outlier, f_trend, f_error = map(_contig, [f, outlier, f_trend, f_error])

    # ffi cast
    _cast = lambda arr: ffi.cast('double *', ffi.from_buffer(arr))
    fs_ffi, ft_ffi, fe_ffi = map(_cast, [f, f_trend, f_error])
    ol_ffi = ffi.cast('bool *', ffi.from_buffer(outlier))

    # run the c++ code
    lib.c_get_trend(fs_ffi, ol_ffi, t.size, kernel_size, ft_ffi, fe_ffi)

    # return trend and stderror in original order
    return f_trend[orig_order], f_error[orig_order]





if __name__=="__main__":
    import matplotlib.pyplot as plt
    from time import time

    # run test
    np.random.seed(0)
    _x = np.linspace(0, 3*np.pi, 10000)
    _y = np.random.normal(loc=np.sin(_x), scale=0.1)
    t0 = time()
    _trend, _error = get_trend(_x, _y, 100)
    o_sig = np.std(_y-_trend)
    print("""\
got trend in {:.0f}ms
input sigma: {:.2f}
output sigma: {:.2f}\
""".format((time()-t0)*1000., 0.1, o_sig))

    # check output is consistent with test validation data
    test_array = np.column_stack((_trend,_error))
    validation_data = np.load("tests/test_validation.npy")
    assert np.count_nonzero(np.abs(test_array-validation_data)>1E-12) == 0

    # if test was successful store a demonstrative figure
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12,12))

    axes[0].scatter(_x, _y, s=1, alpha=0.1)
    axes[0].set_ylabel(r"sin(x) + $\xi$")

    axes[1].scatter(_x, _trend, s=1, alpha=0.1)
    axes[1].set_ylabel(r"trend")

    axes[2].scatter(_x, _error, s=1, alpha=0.1)
    axes[2].set_ylabel(r"$\sigma_{\rm trend}$")

    axes[3].scatter(_x, _y-_trend, s=1, alpha=0.1)
    axes[3].set_xlabel("x")
    axes[3].set_ylabel(r"sin(x) + $\xi$ - trend")

    plt.tight_layout()
    plt.savefig("tests/get_trend_test.png", bbox_inches="tight")
    print("Test successful, demo figure stored as tests/get_trend_test.png")
