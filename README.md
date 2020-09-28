# PLATO_ring_sim
A simulator of ringed planet transits in the context of the PLATO mission.

## Ringed planet transit modelling
The code responsible for simulating the transits of ringed planets is
`ringed_planet_transit.py`. Its only dependency is `numpy`.

The code is demonstrated in `rpt_demo.ipynb`. I compare an rpt non-ringed planet
transit to the Mandel & Agol (2002) planetary transit method as implemented by
PSLS (see below), which requires `transit.py` supplied as part of the PSLS
package.

## PLATO light curve simulation
I rely on the Plato Solar-like Light-curve Simulator `PSLS.py` to incorporate
realistic observational noise and stellar surface variability for testing
purposes. See:<br>
https://sites.lesia.obspm.fr/psls/<br>
and Samadi, R et al. (2019A&A...624A.117S)

This is demonstrated in `rpt_plato.ipynb` and requires the compilation of a
small c++ routine to perform first order smoothing, see below for details.

#### Light curve smoothing
A first-order light curve smoothing is performed with a small c++ routine to
measure a moving window mean and standard error. `get_trend.py` requires
compilation of the c++ code `get_trend.cpp`. This can be done using the included Makefile, and it has no noteworthy dependencies.

`make` builds the library<br>
`make test` tests the library after compilation<br>
`make clean` removes the compiled libraries

n.b. compilation has only been tested on a Debian 10 system using gcc version
8.3.0.
