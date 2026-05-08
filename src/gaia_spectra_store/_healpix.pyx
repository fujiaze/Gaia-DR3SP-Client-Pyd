# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
HEALPix scalar math — Cython-optimized for nside=4 astrometry.

This module provides the scalar HEALPix functions used by the spatial index
query engine.  The vectorized _ang2pix_nest4 lives in xpsd_client.py and
uses NumPy (already C speed).
"""
import math
cimport cython

cdef double PI = math.pi
cdef double DEG2RAD = PI / 180.0
cdef double RAD2DEG = 180.0 / PI


@cython.cfunc
cdef inline double _rad(double deg) noexcept nogil:
    return deg * DEG2RAD


@cython.cfunc
cdef inline int _xyf_to_nest(int nside, int ix, int iy, int face_num) noexcept:
    cdef int npface = nside * nside
    cdef int ix2 = ix, iy2 = iy, scale = 1, pixel = 0
    cdef int nbits = 2
    while nside >> nbits:
        nbits += 1
    nbits -= 1
    for _ in range(nbits):
        pixel += ((ix2 & 1) + ((iy2 & 1) << 1)) * scale
        scale <<= 2
        ix2 >>= 1
        iy2 >>= 1
    pixel += face_num * npface
    return pixel


cpdef int ang2pix(int nside, double ra, double dec, bint lonlat=True, bint nest=True):
    """
    Convert equatorial (ra, dec) in degrees to HEALPix pixel index.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2, e.g. 4).
    ra, dec : float
        Right ascension / declination in degrees.
    lonlat : bool
        If True (default), ra/dec are longitude/latitude in degrees.
    nest : bool
        If True (default), return NESTED ordering.

    Returns
    -------
    int
        HEALPix pixel index.
    """
    cdef double theta, phi, z, abs_z
    cdef double temp1, temp2, tp, sigma, x
    cdef int jp, jm, ifp, ifm, face_num, ix, iy, ntt

    if lonlat:
        theta = _rad(90.0 - dec)
        phi = _rad(ra)
    else:
        theta = ra
        phi = dec

    z = math.cos(theta)
    abs_z = z if z >= 0 else -z

    if abs_z <= (2.0 / 3.0):
        temp1 = nside * (0.5 + (phi * 0.75 / PI))
        temp2 = nside * z * 0.75
        jp = <int>(temp1 - temp2)
        jm = <int>(temp1 + temp2)
        ifp = jp // nside
        ifm = jm // nside

        if ifp == ifm:
            face_num = 4 if ifp < 0 else 0
        elif ifp < ifm:
            face_num = 2 if ifp < 0 else 6
        else:
            face_num = 4 if ifm < 0 else 8
        face_num = face_num % 12

        ix = jm % nside
        iy = (nside - 1) - (jp % nside)
    else:
        ntt = <int>(phi * 2.0 / PI)
        tp = phi - ntt * PI / 2.0
        if abs_z < 0.99:
            sigma = 2.0 * (1.0 - abs_z)
        else:
            sigma = (math.sin(theta) * math.sin(theta)) / abs_z
        x = nside * sigma * 0.5
        ix = <int>x
        iy = ix

        if abs_z > 0.99:
            jp = <int>(tp * x)
            jm = <int>((PI / 2.0 - tp) * x)
        else:
            jp = <int>((tp / PI * 0.5 + 0.5) * nside * sigma)
            jm = <int>(((1.0 - tp / PI * 0.5 - 0.5) + 0.5) * nside * sigma)

        if ix >= nside: ix = nside - 1
        if jp >= nside: jp = nside - 1
        if jm >= nside: jm = nside - 1

        if z > 0:
            if ntt == 0:      face_num = 0
            elif ntt == 1:    face_num = 2
            elif ntt == 2:    face_num = 4
            else:             face_num = 6
            ix = nside - 1 - ix
        else:
            if ntt == 0:      face_num = 8
            elif ntt == 1:    face_num = 10
            elif ntt == 2:    face_num = 12
            else:             face_num = 14

    if nest:
        return _xyf_to_nest(nside, ix, iy, face_num % 12)
    return face_num * nside * nside + iy * nside + ix


cpdef list query_disc(int nside, tuple vec, double radius, bint nest=True, bint inclusive=True):
    cdef int npix = 12 * nside * nside
    cdef list pixels = []
    cdef int i
    cdef double pix_ra, pix_dec, dot, ang_dist
    cdef double vx = vec[0], vy = vec[1], vz = vec[2]

    for i in range(npix):
        pix_theta, pix_phi = pix2ang(nside, i, nest=True, lonlat=False)
        dot = math.sin(pix_theta) * math.cos(pix_phi) * vx + \
              math.sin(pix_theta) * math.sin(pix_phi) * vy + \
              math.cos(pix_theta) * vz
        if dot > 1.0: dot = 1.0
        if dot < -1.0: dot = -1.0
        ang_dist = math.acos(dot)
        if inclusive:
            if ang_dist <= radius:
                pixels.append(i)
        else:
            if ang_dist < radius:
                pixels.append(i)
    return pixels


cpdef tuple pix2ang(int nside, int ipix, bint nest=True, bint lonlat=False):
    cdef int npface = nside * nside
    cdef int face_num, ipf, ix = 0, iy = 0, scale = 1, _, ipf_low
    cdef int nbits = 2
    while nside >> nbits:
        nbits += 1
    nbits -= 1

    if nest:
        face_num = ipix // npface
        ipf = ipix % npface
        for _ in range(nbits):
            ipf_low = ipf & 3
            ix += scale * (ipf_low & 1)
            iy += scale * ((ipf_low >> 1) & 1)
            scale <<= 1
            ipf >>= 2
    else:
        face_num = ipix // npface
        ipf = ipix % npface
        ix = ipf // nside
        iy = ipf % nside

    cdef int jr = (iy << nbits) + ix
    cdef double z, phi, theta
    cdef double phi_cap

    if face_num < 4:
        phi = (jr + 0.5) / nside * PI / 2.0 + face_num * PI / 2.0
        z = (2.0 * nside - jr - 1.0) / (1.5 * nside)
    elif face_num < 8:
        z = 1.0 - jr * jr / (3.0 * nside * nside)
        phi_cap = (face_num - 4) * PI / 2.0
        phi = phi_cap + (0.5 * (jr + 0.5 - ix * nside) * PI / 2.0 / nside if jr < nside else 0)
    else:
        z = -1.0 + jr * jr / (3.0 * nside * nside)
        phi_cap = (face_num - 8) * PI / 2.0
        phi = phi_cap + (0.5 * (jr + 0.5 - ix * nside) * PI / 2.0 / nside if jr < nside else 0)

    if z > 1.0: z = 1.0
    if z < -1.0: z = -1.0
    theta = math.acos(z)

    if lonlat:
        return phi * RAD2DEG, 90.0 - theta * RAD2DEG
    return theta, phi
