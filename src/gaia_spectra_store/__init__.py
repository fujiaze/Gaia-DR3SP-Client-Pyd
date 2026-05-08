"""
Gaia DR3/SP XPSD Direct Client
================================

High-performance native star catalog query engine with BP/RP spectrum retrieval.

Directly reads PixInsight XPSD format files — no conversion needed.
Supports 2.19亿 stars across 20 files with HEALPix spatial index.

Usage::

    from gaia_spectra_store import GaiaClient

    client = GaiaClient("GaiaDR3SP/", "gaia_indices/")

    # First run: build spatial index (6s, once)
    client.build_index()

    # Cone search (21ms for 1000 stars across all 20 files)
    stars = client.cone_search(ra=180, dec=0, radius_deg=5)

    # Single-star spectrum retrieval (2ms)
    spec = client.get_spectrum(source_id=1000000)
    print(spec["bp_spectrum"])       # float64[343] W·m⁻²·nm⁻¹
    print(spec["wavelength"][0])     # 336.0 nm
    print(spec["wavelength"][-1])    # 1020.0 nm
"""
import math
import msgpack
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .xpsd_client import (
    XPSDFile,
    EncodedStarData,
    EncodedStarSPData,
    _ang2pix_nest4,
    WAVELENGTH_START,
    WAVELENGTH_STEP,
    WAVELENGTH_COUNT,
    STAR_STRIDE,
)
from .index_builder import build_all_indices

_WAVELENGTH = np.arange(WAVELENGTH_START,
                         WAVELENGTH_START + WAVELENGTH_COUNT * WAVELENGTH_STEP,
                         WAVELENGTH_STEP, dtype=np.float64)

_STAR_STRIDE = STAR_STRIDE
_WL_COUNT = WAVELENGTH_COUNT


def _decode_leaf_vectorized(data, leaf, proj, cra, cdec):
    """向量化解码一个叶节点：NumPy strides 一次读取所有星（无 Python 循环）。"""
    n = len(data) // _STAR_STRIDE
    if n == 0:
        return [], [], [], [], [], [], [], [], [], []

    buf = data
    ss = _STAR_STRIDE
    dx = np.ndarray(n, np.uint32, buf, 0, (ss,)).astype(np.float64)
    dy = np.ndarray(n, np.uint32, buf, 4, (ss,)).astype(np.float64)
    pxv = np.ndarray(n, np.float32, buf, 8, (ss,))
    pmrav = np.ndarray(n, np.float32, buf, 12, (ss,))
    pmdecv = np.ndarray(n, np.float32, buf, 16, (ss,))
    mg = np.ndarray(n, np.uint16, buf, 20, (ss,)).astype(np.float32) * 0.001 - 1.5
    mbp = np.ndarray(n, np.uint16, buf, 22, (ss,)).astype(np.float32) * 0.001 - 1.5
    mrp = np.ndarray(n, np.uint16, buf, 24, (ss,)).astype(np.float32) * 0.001 - 1.5
    dra = np.ndarray(n, np.int16, buf, 26, (ss,)).astype(np.float64)
    flg = np.ndarray(n, np.int32, buf, 28, (ss,))
    fmin = np.ndarray(n, np.float32, buf, 32, (ss,))
    fmul = np.ndarray(n, np.float32, buf, 36, (ss,))
    flux = np.ndarray((n, _WL_COUNT), np.uint8, buf, 40, (ss, 1)).copy()

    x0, y0 = leaf['x0'], leaf['y0']
    x = x0 + dx / (3600.0 * 1000.0 * 500.0)
    y = y0 + dy / (3600.0 * 1000.0 * 500.0)

    if proj == 'Equirectangular':
        ra = x + cra
        dec = y
    elif proj == 'AzimuthalEquidistant':
        xr = np.radians(x); yr = np.radians(y)
        ra = np.degrees(np.arctan2(xr, yr))
        ra[ra < 0] += 360.0
        dec = np.degrees(np.arcsin(np.cos(np.sqrt(xr*xr + yr*yr))))
        if cdec < 0:
            dec = -dec
    else:
        ra, dec = x, y

    ndra = dra != 0
    if np.any(ndra):
        ra = np.where(ndra, ra + dra / (3600.0 * 1000.0 * 100.0), ra)
    ra %= 360.0

    return (ra, dec, mg, mbp, mrp, pxv, pmrav, pmdecv, flg, fmin, fmul, flux)


def _great_circle_filter(ra_stars, dec_stars, ra_center, dec_center, radius_rad):
    """向量化大圆距离过滤。返回布尔掩码。"""
    sin_d0 = math.sin(math.radians(dec_center))
    cos_d0 = math.cos(math.radians(dec_center))

    dec_rad = np.radians(dec_stars)
    sin_ds = np.sin(dec_rad)
    cos_ds = np.cos(dec_rad)

    d_ra = np.radians(ra_stars - ra_center)
    d_ang = sin_d0 * sin_ds + cos_d0 * cos_ds * np.cos(d_ra)
    d_ang = np.clip(d_ang, -1.0, 1.0)
    return np.arccos(d_ang) <= radius_rad


class GaiaClient:
    """Gaia DR3/SP XPSD 直接查询客户端。

    零膨胀 — 直接读取 XPSD 文件，使用外挂 HEALPix 索引实现毫秒级查询。

    Parameters
    ----------
    data_dir : str or Path
        XPSD 文件目录，包含 gdr3sp-1.0.0-*.xpsd 文件。
    index_dir : str or Path, optional
        外挂索引目录。默认与 data_dir 相同。

    Examples
    --------
    >>> client = GaiaClient("GaiaDR3SP/")
    >>> client.build_index()
    >>> stars = client.cone_search(180, 0, 5, limit=100)
    >>> spec = client.get_spectrum(1000000)
    """
    def __init__(self, data_dir, index_dir=None):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir) if index_dir else self.data_dir
        self._files = {}
        self._indices = {}
        self._hp_to_leaves = {}
        self._source_id_map = {}
        self._load()

    @property
    def stats(self):
        return {
            'files': len(self._files),
            'sources': sum(f.total_sources for f in self._files.values()),
            'indexed': len(self._hp_to_leaves) > 0,
        }

    def _load(self):
        xpsd_files = sorted(self.data_dir.glob('*.xpsd'))
        if not xpsd_files:
            raise FileNotFoundError(f"No .xpsd files in {self.data_dir}")

        for xp in xpsd_files:
            key = xp.stem
            self._files[key] = XPSDFile(xp)
            xpi = self.index_dir / xp.with_suffix('.xpsi').name
            if xpi.exists():
                raw = xpi.read_bytes()
                self._indices[key] = msgpack.unpackb(raw, raw=False, strict_map_key=False)

        self._init_index()

    def _init_index(self):
        self._hp_to_leaves = {}
        for key, idx in self._indices.items():
            for pix_str, leaf_indices in idx.get('hp_index', {}).items():
                pix = int(pix_str)
                if pix not in self._hp_to_leaves:
                    self._hp_to_leaves[pix] = []
                for li in leaf_indices:
                    if li < len(idx['leaf_bbox']):
                        self._hp_to_leaves[pix].append((key, li))

        offset = 0
        for key in sorted(self._files.keys()):
            f = self._files[key]
            self._source_id_map[key] = (offset, offset + f.total_sources - 1)
            offset += f.total_sources

    def build_index(self):
        """构建或重建外挂空间索引（首次使用必须调用）。"""
        n = build_all_indices(str(self.data_dir), str(self.index_dir))
        self._load()
        return n

    def cone_search(self, ra, dec, radius_deg,
                    mag_low=-1.5, mag_high=26, limit=None):
        """锥形搜索。返回搜索半径内全部恒星（无 limit 限制）。

        Parameters
        ----------
        ra, dec : float
            搜索中心坐标（度）。
        radius_deg : float
            搜索半径（度）。
        mag_low, mag_high : float
            星等范围。
        limit : int or None
            最大返回数。None = 不限。

        Returns
        -------
        list of EncodedStarSPData
        """
        target_pixels = set()
        for pix in _query_disc_pixels(4, ra, dec, radius_deg):
            target_pixels.add(pix)
        target_pixels.update(int(p) for p in _ang2pix_nest4(
            np.array([ra], dtype=np.float64), np.array([dec], dtype=np.float64)))

        candidates = []
        seen = set()
        for pix in target_pixels:
            if pix not in self._hp_to_leaves:
                continue
            for key, li in self._hp_to_leaves[pix]:
                uid = (key, li)
                if uid in seen:
                    continue
                seen.add(uid)
                leaf = self._indices[key]['leaf_bbox'][li]
                if _bbox_intersects(ra, dec, radius_deg,
                                     leaf['ra_min'], leaf['ra_max'],
                                     leaf['dec_min'], leaf['dec_max']):
                    candidates.append((key, leaf))

        results = []
        radius_rad = math.radians(radius_deg)

        for key, leaf in candidates:
            if limit is not None and len(results) >= limit:
                break
            f = self._files[key]
            data = f.read_leaf_block(leaf['block_offset'],
                                      leaf['compressed_size'],
                                      leaf['block_size'])
            if data is None:
                continue

            proj = f.trees[leaf['tree_idx']]['config']['projection']
            cra = f.trees[leaf['tree_idx']]['config']['center_ra']
            cdec = f.trees[leaf['tree_idx']]['config']['center_dec']

            (s_ra, s_dec, mg, mbp, mrp,
             pxv, pmrav, pmdecv, flg, fmin, fmul, flux) = \
                _decode_leaf_vectorized(data, leaf, proj, cra, cdec)

            if len(s_ra) == 0:
                continue

            # Bulk magnitude filter
            mag_ok = (mg >= mag_low) & (mg <= mag_high)

            # Bulk great-circle distance
            gc_ok = _great_circle_filter(s_ra, s_dec, ra, dec, radius_rad)

            mask = mag_ok & gc_ok
            idx = np.flatnonzero(mask)

            for ii in idx:
                sp = EncodedStarSPData.__new__(EncodedStarSPData)
                sp._decoded_ra = float(s_ra[ii])
                sp._decoded_dec = float(s_dec[ii])
                sp.magG = float(mg[ii])
                sp.magBP = float(mbp[ii])
                sp.magRP = float(mrp[ii])
                sp.parx = float(pxv[ii])
                sp.pmra = float(pmrav[ii])
                sp.pmdec = float(pmdecv[ii])
                sp.flags = int(flg[ii])
                sp.flux_min = float(fmin[ii])
                sp.flux_mul = float(fmul[ii])
                sp.flux = flux[ii].copy()
                sp.dx = 0; sp.dy = 0; sp.dra = 0
                results.append(sp)
                if limit is not None and len(results) >= limit:
                    break
        return results

    def get_spectrum(self, source_id, normalize=False, photon_flux=False):
        """按全局编号查询单星光谱。

        Parameters
        ----------
        source_id : int
            全局恒星编号（0 ~ 219165265）。
        normalize : bool
            归一化到 [0,1]。
        photon_flux : bool
            转换为光子通量单位。

        Returns
        -------
        dict or None
            source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag,
            phot_rp_mean_mag, parallax, pmra, pmdec, flags,
            bp_spectrum, rp_spectrum, wavelength
        """
        for key, (smin, smax) in self._source_id_map.items():
            if source_id < smin or source_id > smax:
                continue
            f = self._files[key]
            local_sid = source_id - smin
            idx = self._indices.get(key)
            if idx is None:
                continue

            leaves_db = idx['leaf_bbox']
            lo, hi = 0, len(leaves_db) - 1
            target_leaf = None
            while lo <= hi:
                mid = (lo + hi) // 2
                leaf = leaves_db[mid]
                if local_sid < leaf['sid_offset']:
                    hi = mid - 1
                elif local_sid >= leaf['sid_offset'] + leaf['star_count']:
                    lo = mid + 1
                else:
                    target_leaf = leaf
                    break
            if target_leaf is None:
                return None

            leaf = target_leaf
            data = f.read_leaf_block(leaf['block_offset'],
                                      leaf['compressed_size'],
                                      leaf['block_size'])
            if data is None:
                return None

            i = local_sid - leaf['sid_offset']
            sp = EncodedStarSPData.decode(data, i * STAR_STRIDE)
            proj = f.trees[leaf['tree_idx']]['config']['projection']
            cra = f.trees[leaf['tree_idx']]['config']['center_ra']
            cdec = f.trees[leaf['tree_idx']]['config']['center_dec']

            x = leaf['x0'] + sp.dx / (3600.0 * 1000.0 * 500.0)
            y = leaf['y0'] + sp.dy / (3600.0 * 1000.0 * 500.0)
            s_ra, s_dec = _unproject(x, y, proj, cra, cdec)
            if sp.dra != 0:
                s_ra += sp.dra / (3600.0 * 1000.0 * 100.0)
            s_ra %= 360.0

            return {
                'source_id': source_id,
                'ra': s_ra,
                'dec': s_dec,
                'phot_g_mean_mag': sp.magG,
                'phot_bp_mean_mag': sp.magBP,
                'phot_rp_mean_mag': sp.magRP,
                'parallax': sp.parx,
                'pmra': sp.pmra,
                'pmdec': sp.pmdec,
                'flags': sp.flags,
                'bp_spectrum': sp.decode_spectrum(normalize, photon_flux),
                'rp_spectrum': sp.decode_spectrum(normalize, photon_flux),
                'wavelength': _WAVELENGTH.copy(),
            }
        return None

    def close(self):
        for f in self._files.values():
            f.close()


def _bbox_intersects(ra, dec, r, ra_min, ra_max, dec_min, dec_max):
    d_ra = max(0, ra_min - ra, ra - ra_max)
    if d_ra > 180: d_ra = 360 - d_ra
    d_dec = max(0, dec_min - dec, dec - dec_max)
    if d_ra < 1e-12 and d_dec < 1e-12:
        return True
    return math.sqrt(d_ra**2 + d_dec**2) <= r * 1.5


def _query_disc_pixels(nside, ra, dec, radius_deg):
    theta = math.radians(90.0 - dec)
    phi = math.radians(ra)
    vx = math.sin(theta) * math.cos(phi)
    vy = math.sin(theta) * math.sin(phi)
    vz = math.cos(theta)
    npix = 12 * nside * nside
    pixels = []
    radius_rad = math.radians(radius_deg)
    for i in range(npix):
        pt, pp = _pix2ang_nest(nside, i)
        dot = math.sin(pt)*math.cos(pp)*vx + math.sin(pt)*math.sin(pp)*vy + math.cos(pt)*vz
        if dot > 1: dot = 1
        if dot < -1: dot = -1
        if math.acos(dot) <= radius_rad:
            pixels.append(i)
    return pixels


def _pix2ang_nest(nside, ipix):
    npface = nside * nside
    face_num = ipix // npface
    ipf = ipix % npface
    ix = 0; iy = 0; scale = 1
    for _ in range(nside.bit_length() - 1):
        ipf_low = ipf & 3
        ix += scale * (ipf_low & 1)
        iy += scale * ((ipf_low >> 1) & 1)
        scale <<= 1
        ipf >>= 2
    jr = (iy << (nside.bit_length() - 1)) + ix
    if face_num < 4:
        phi = (jr + 0.5) / nside * math.pi / 2.0 + face_num * math.pi / 2.0
        z = (2.0 * nside - jr - 1.0) / (1.5 * nside)
    elif face_num < 8:
        z = 1.0 - jr*jr / (3.0 * nside*nside)
        phi_cap = (face_num - 4) * math.pi / 2.0
        phi = phi_cap + (0.5*(jr+0.5-ix*nside)*math.pi/2.0/nside if jr < nside else 0)
    else:
        z = -1.0 + jr*jr / (3.0 * nside*nside)
        phi_cap = (face_num - 8) * math.pi / 2.0
        phi = phi_cap + (0.5*(jr+0.5-ix*nside)*math.pi/2.0/nside if jr < nside else 0)
    if z > 1: z = 1
    if z < -1: z = -1
    return math.acos(z), phi


def _unproject(x, y, projection, center_ra, center_dec):
    if projection == 'Equirectangular':
        return x + center_ra, y
    elif projection == 'AzimuthalEquidistant':
        xr = math.radians(x); yr = math.radians(y)
        ra = math.degrees(math.atan2(xr, yr))
        if ra < 0: ra += 360
        dec = math.degrees(math.asin(math.cos(math.sqrt(xr*xr + yr*yr))))
        if center_dec < 0: dec = -dec
        return ra, dec
    return x, y
