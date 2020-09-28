#!/usr/bin/env python3

import numpy as np


__version__ = "0.1"


def quad_limb_dark(r_rs, a, b):
    """Intensity as a function of radius for the quadratic limb darkening model.

    Parameters:
    r_rs : ndarray
        Radii as a fraction of stellar radius.
    a,b, : float, float
        Quadratic limb darkening coefficients.

    Returns:
    I : ndarray
        Stellar intensity at the requested radii.
    """
    Ir = np.zeros(r_rs.shape, dtype=np.float64)
    lt1 = np.abs(r_rs)<=1.0
    mu = np.cos(np.arcsin(r_rs[lt1]))
    Ir[lt1] = (1 - a * (1 - mu) - b * (1 - mu)**2) / (1 - a/3 - b/6) / np.pi
    return Ir



def reflect_2d(array, multiplier=None):
    """Reflect an array about both cartesian axes, optionally include a
    multiplier.

    Parameters:
    array : ndarray
        The array to reflect, size (N,M).
    multiplier : ndarray of size=4; optional
        This is an array of values by which to multiply each reflected segment.
        The array is flattened before use. This is most useful for cartesian
        coordinate arrays. (Default None.)

    Returns:
    output : ndarray
        The input array reflected about both cartesian axes with new size
        (2N,2M).
    """

    if multiplier is not None:
        m = np.array(multiplier).flatten()
    else:
        m = np.ones(4, dtype=np.int)

    return np.block([[m[0]*np.flip(array), m[1]*np.flip(array,axis=0)],
                     [m[2]*np.flip(array,axis=1), m[3]*array]
                    ])




def build_ringed_occulter(rings=None,
                          gamma=0.0,
                          Ngrid=100,
                          super_sample_factor=10,
                          full_output=False,
                         ):
    """ Build a greyscale transparency image of a ringed occulter.

    Parameters:
    rings : N, list of 3, tuples; optional
        The list of N rings to include in the model. Each list element is a 3
        component tuple of (i_r, o_r, T), where i_r and o_r are the ring inner
        and outer radii in units of planet radii and T is the ring transparency
        in the range [0,1]. T=0 is fully opaque, T=1 is completely transparent.
        (Default None, i.e. no rings.)
    gamma : float; optional
        Inclination angle relative to the line of sight to the observer in
        radians. (Default 0.0.)
    Ngrid : int; optional
        The number of grid points (or pixels if you like) in the occulter image
        along the major axis per planet radius. The actual number of grid points
        returned depends on maximum outer radius of the rings. The number of
        grid points in the minor dimension depends on the inclination angle,
        gamma, but the minimum value will be 2*Ngrid due to the spherical planet
        assumption. Another way to think about it is that Ngrid is the desired
        radius of the planet in grid steps. (Default 100.)
    super_sample_factor : int; optional
        Super-sample edge grid elements by this factor. (Default 10.)
    full_output : bool; optional
        If True return the complete 2-dimensional array (image) of the occulter.
        If False return only the array elements which are occulted, i.e. have
        T<1. (Default False.)

    Returns:
    profile : ndarray
        The transparency profile of the occulter. If full_output is True this is
        an array with ndim=2, otherwise it is flattened and includes only
        elements where T<1.
    x, y : ndarrays
        The x and y positions of each profile element in units of planet radii
        and centred at the planet centre. These are either ndim=2 or flattened
        arrays depending on the state of full_output.
        Hint: The occulter can be rescaled, rotated, translated, etc. by simply
        applying the relevant transformations to these arrays.
    area : float
        The area of each transparency profile element in units of planetary
        radii squared.

    TODO:
        - Run line-profiler to see where the bottlenecks are.
        - This function could also be much faster if ported to c++.
    """

    # max ring size
    if rings is not None:
        # largest outer radius of rings in planetary radii
        o_r_max = np.max([r[1] for r in rings])
    else:
        # no rings, max size is the planet radius
        o_r_max = 1.0

    ### build the profile and x,y arrays ###
    # 'half-grid' size in major and minor dimensions
    # 'half-grid' since we're dealing in planetary radii, and we can assume
    # symmetry in both dimensions for this simple example so there's no need
    # to evaluate the whole thing
    Ngrid_maj = int(np.ceil(o_r_max*Ngrid))
    Ngrid_min = max(Ngrid, int(np.ceil(Ngrid_maj*np.sin(gamma))))
    # initialise the arrays (translate xgrid, ygrid later)
    pgrid = np.ones((Ngrid_maj, Ngrid_min), dtype=np.float64)
    xgrid, ygrid = np.mgrid[0:Ngrid_maj, 0:Ngrid_min]


    # add the planet
    irad = np.sqrt(xgrid**2 + ygrid**2) # inner radius of each pixel
    orad = np.sqrt((xgrid+1)**2 + (ygrid+1)**2) # outer radius of each pixel
    # transparency is zero where pixel is fully inside planet
    pgrid[orad<=Ngrid] *= 0.0
    # transparency at the limits of the circle is the fraction of each grid
    # point not covered by the circle. I'll use super-sampling for now, but
    # ultimately it would likely be faster to calculate this analytically
    ss_frac = 1./super_sample_factor
    sup_x, sup_y = map(lambda a: ss_frac*(a+0.5),
                       np.mgrid[0:super_sample_factor,0:super_sample_factor])
    # +0.5 so that sup_x,sup_y are mid points of super-sample grid
    fracs = np.where((irad<Ngrid) & (orad>Ngrid))
    for _x,_y in zip(*fracs):
        pgrid[_x,_y] *= ss_frac**2 * np.count_nonzero(
            np.sqrt((_x+sup_x)**2 + (_y+sup_y)**2) > Ngrid
        )


    # add the rings
    if rings is not None and gamma>0.0:
        for i_r, o_r, T in rings: # inner rad, outer rad, transparency
            # major and minor axis sizes for the inner and outer edges
            rmaj_i, rmin_i = i_r*Ngrid, i_r*np.sin(gamma)*Ngrid
            rmaj_o, rmin_o = o_r*Ngrid, o_r*np.sin(gamma)*Ngrid
            # where the grid elements are fully inside a ring
            inring = ( (( (xgrid)  / rmaj_i)**2 + ( (ygrid)  / rmin_i)**2 >= 1)
                     & (((xgrid+1) / rmaj_o)**2 + ((ygrid+1) / rmin_o)**2 <= 1))
            pgrid[inring] *= T
            # where the grid elements are partly inside a ring
            # pixel is fractionally covered on inner edge
            inner_lim = (
               (( (xgrid)  / rmaj_i)**2 + ( (ygrid)  / rmin_i)**2 <=  1)
             &
               (((xgrid+1) / rmaj_i)**2 + ((ygrid+1) / rmin_i)**2 >=  1)
            )
            # pixel is fractionally covered on outer edge
            outer_lim = (
               (( (xgrid)  / rmaj_o)**2 + ( (ygrid)  / rmin_o)**2 <=  1)
             &
               (((xgrid+1) / rmaj_o)**2 + ((ygrid+1) / rmin_o)**2 >=  1)
            )
            # also no need to evaluate where the ring is in front of or behind
            # the planet
            fracs = np.where((inner_lim | outer_lim) & (irad>=Ngrid-1))
            # evaluate fractional coverage with super-sampling again
            for _x,_y in zip(*fracs):
                # count of super samples outside ring
                count_out = np.count_nonzero(
                     (((_x+sup_x) / rmaj_i)**2 + ((_y+sup_y) / rmin_i)**2 > 1)
                    &(((_x+sup_x) / rmaj_o)**2 + ((_y+sup_y) / rmin_o)**2 < 1)
                )
                # evaluate contribution of rings to partially filled elements
                pgrid[_x,_y] *= count_out * ss_frac**2 * (T-1) + 1



    # increment grid positions by half a step to be centred on each element and
    # normalise to planet radius, then reflect in both axes to generate full
    # image
    xgrid = reflect_2d((xgrid+0.5)/Ngrid, multiplier=[[-1,-1],[1,1]])
    ygrid = reflect_2d((ygrid+0.5)/Ngrid, multiplier=[[-1,1],[-1,1]])
    pgrid = reflect_2d(pgrid)

    # calculate area of each grid element
    A_elem = (1./Ngrid)**2 # units of r_planet^2

    if full_output:
        return pgrid, xgrid, ygrid, A_elem
    else:
        ret = pgrid<1.0
        return pgrid[ret], xgrid[ret], ygrid[ret], A_elem



def occult_star(transparency_mask,
                planet_radius,
                offset_x,
                offset_y,
                obliquity,
                ld_params):
    """ Produce a stellar flux sequence during occultation by an object with a
    given transparency mask.

    Parameters:
    transparency_mask : tuple of len=4
        A tuple of the transparency values, x positions and y positions of each
        transparency element, and the area of each transparency element in units
        of planetary radii squared. This tuple is as returned by the
        build_ringed_occulter function.
    planet_radius : float
        The radius of the planet as a fraction of the radius of the star.
    offset_x : ndarray of size N
        This is an array of N positional offsets of the planet centre from the
        planet position at the instant of inferior conjunction in units of
        stellar radii. In contrast to standard spherical planet transit
        modelling the sign on offset_x is important wherever the obliquity is
        non-zero.
    offset_y : float
        This is the on-sky position of the planet centre relative to the stellar
        centre in units of stellar radii at the instant of inferior conjunction,
        i.e. the minimum on-sky separation but with the sign preserved, making
        it distinct from the planetary transit impact parameter. In principle
        the sign on offset_y is only important whenever the obliquity is
        non-zero, otherwise it can be treated exactly as the impact parameter.
    obliquity : float
        The axial tilt/axial inclination/obliquity of the planet in radians. In
        reality this defines the angle between the projected major axis of the
        planetary ring system and it's direction of motion. Where the ring
        system lies on the planetary equator this is the same angle as the
        obliquity of the planet. The direction of motion is 'anticlockwise'
        assuming offset_x increases from left to right and offset_y increases
        from bottom to top.
    ld_params : array of length 2
        An array of 2 elements corresponding to the stellar quadratic limb
        darkening coefficients.

    Returns:
    light_curve : ndarray of size N
        An array the same length as the positional_offsets array containing
        floating point values corresponding to the fraction of stellar flux
        remaining after subtracting that obscured by the transiting body.

    TODO:
        - Run line-profiler to see where the bottlenecks are.
        - This function might be a bit faster if ported to c++.
        - Give the option to evaluate serially. The vectorized version is fast
        but relatively memory intensive, which can be a problem where a dense
        occulter transparency grid and a large offset_x array is used. Porting
        to c++ might help retain the speed without requiring a lot of memory.
        - This function can also be trivially parallelised with multiprocessing,
        maybe this could be given as an option. Porting to c++ might render it
        unnecessary though.
        - This function might also benefit from the use of sparse matrices.
    """

    # break out transparency mask contents
    p,x,y,A = transparency_mask

    # scale the transparency mask size
    _x,_y = map(lambda _a : _a*planet_radius, [x,y])
    # scale the transparency mask element area
    _A = A*planet_radius**2

    # rotate the transparency mask by the obliquity
    c_oblq, s_oblq = np.cos(obliquity), np.sin(obliquity)
    _x = _x*c_oblq - _y*s_oblq
    _y = _x*s_oblq + _y*c_oblq

    # apply the positional offsets
    _y += offset_y
    _x = _x+offset_x[:,None]

    # where no occultation occurs return an array of ones
    if np.min(np.sqrt(_x**2+_y**2))>1.0:
        return np.ones(offset_x.size, dtype=np.float64)

    # evaluate stellar flux at the positions
    I = quad_limb_dark(np.sqrt(_x**2+_y**2), *ld_params) * _A

    # return the flux relative to baseline
    return 1 + ( np.sum(I*p[None,:], axis=1) - np.sum(I, axis=1) )
