#!/usr/bin/env python3

import numpy as np
from cffi import FFI

ffi = FFI()
ffi.cdef("""
void c_get_trend(const double *t,
                 const double *f,
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

    TODO:
        - Fix minor issue of kernel size tweak where clipping points if this
          function ever goes into production. Otherwise it's fine.
    """

    # deal with masking data
    if data_mask is not None:
        t_sorted = t.copy()[data_mask]
        f_sorted = f.copy()[data_mask]
    else:
        t_sorted = t.copy()
        f_sorted = f.copy()

    # sort the data chronologically
    order = np.argsort(t_sorted)
    f_sorted = f_sorted[order]
    t_sorted = t_sorted[order]

    # remove clear outliers (e.g. transit points)
    # technically I should account for these in the kernel size, but so few
    # should be removed that I very much doubt this is a problem. Fix if this
    # ever goes into production though.
    if sigma_clip:
        for level in [10,9,8,7,6,5]: # reduce cut level each time
            stdev = np.std(f_sorted)
            not_outlier = f_sorted < level*stdev
            t_sorted, f_sorted = t_sorted[not_outlier], f_sorted[not_outlier]

    # initialise trend arrays
    f_trend = f_sorted.copy()
    f_error = np.zeros(f_sorted.size, dtype=np.float64)

    # make them contiguous
    _contig = lambda arr: np.ascontiguousarray(arr)
    t_sorted, f_sorted, f_trend, f_error = map(_contig,
        [t_sorted, f_sorted, f_trend, f_error])

    # ffi cast
    _cast = lambda arr: ffi.cast('double *', ffi.from_buffer(arr))
    ts_ffi, fs_ffi, ft_ffi, fe_ffi = map(_cast,
        [t_sorted, f_sorted, f_trend, f_error])

    # run the c++ code
    lib.c_get_trend(ts_ffi, fs_ffi, t_sorted.size, kernel_size, ft_ffi, fe_ffi)

    # Now interpolate over any clipped points. This has the added benefits of:
    # - returning the output arrays to the same order as the input array
    #   (although there are faster ways to do this).
    # - Extrapolating the first and last points if they were clipped. Since the
    #   trend is generally very smooth this should be fine.
    trend = f*np.nan
    error = f*np.nan
    # find nearest indices
    i1 = np.searchsorted(t_sorted, t, side='right')
    i0 = i1-1
    # deal with an uncomfortable bounds error
    i0[i1==t_sorted.size] -= 1
    i1[i1==t_sorted.size] -= 1
    # values at each index
    y0, y1 = f_trend[i0], f_trend[i1]
    e0, e1 = f_error[i0], f_error[i1]
    # distance from each point
    dx0, dx1 = t - t_sorted[i0], t_sorted[i1] - t
    dx = t_sorted[i1] - t_sorted[i0]
    # where there are multiple identical t values (e.g. the time series is phase
    # folded), we'll have to work around that
    dxnz = dx!=0
    # interpolated values
    trend[dxnz] = (y0*dx1 + y1*dx0)[dx>0] / dx[dxnz]
    error[dxnz] = np.sqrt((e0*dx1)**2 + (e1*dx0)**2)[dxnz] / dx[dxnz]
    # where dx is zero just use the nearest value
    trend[~dxnz] = y0[~dxnz]
    error[~dxnz] = e0[~dxnz]

    return trend, error





if __name__=="__main__":
    import matplotlib.pyplot as plt
    from time import time

    # run the code
    test_data = np.load("tests/get_trend_test_data.npy")
    t0 = time()
    _trend, _error = get_trend(test_data["time"], test_data["dflux_ppm"], 200)
    print("got trend in {:.3f}s".format(time()-t0))

    # check output is consistent with test validation data
    test_array = np.column_stack((_trend,_error))
    validation_data = np.load("tests/test_validation.npy")
    assert np.count_nonzero(test_array-validation_data) == 0

    # if test was successful store a demonstrative figure
    plt.figure(figsize=(12,12))

    plt.subplot(411)
    plt.scatter(test_data["time"], test_data["dflux_ppm"], s=1, alpha=0.05)
    plt.xlabel("t (s)")
    plt.ylabel(r"$\Delta$ flux (ppm)")
    plt.title("before")

    plt.subplot(412)
    plt.scatter(test_data["time"], _trend, s=1, alpha=0.05)
    plt.xlabel("t (s)")
    plt.ylabel(r"$\Delta$ flux (ppm)")
    plt.title("trend")

    plt.subplot(413)
    plt.scatter(test_data["time"], _error, s=1, alpha=0.05)
    plt.xlabel("t (s)")
    plt.ylabel(r"$\Delta$ flux (ppm)")
    plt.title(r"$\sigma_{\rm trend}$")

    plt.subplot(414)
    plt.scatter(test_data["time"], test_data["dflux_ppm"]-_trend, s=1, alpha=0.05)
    plt.xlabel("t (s)")
    plt.ylabel(r"$\Delta$ flux (ppm)")
    plt.title("after")

    plt.tight_layout()
    plt.savefig("tests/get_trend_test.png", bbox_inches="tight")
    print("Test successful, demo figure stored as tests/get_trend_test.png")
