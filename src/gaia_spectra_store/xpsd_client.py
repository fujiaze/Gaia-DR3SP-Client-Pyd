import struct
import zlib
import mmap
import os
import math
import bisect
import xml.etree.ElementTree as ET
from pathlib import Path
from functools import lru_cache

import numpy as np

WAVELENGTH_START = 336.0
WAVELENGTH_STEP = 2.0
WAVELENGTH_COUNT = 343
STAR_STRIDE = 40 + (WAVELENGTH_COUNT + (WAVELENGTH_COUNT & 1))

_WAVELENGTH = np.arange(WAVELENGTH_START,
                         WAVELENGTH_START + WAVELENGTH_COUNT * WAVELENGTH_STEP,
                         WAVELENGTH_STEP, dtype=np.float64)


class EncodedStarData:
    __slots__ = ('dx', 'dy', 'parx', 'pmra', 'pmdec', 'magG', 'magBP', 'magRP',
                 'dra', 'flags', '_decoded_ra', '_decoded_dec')
    SIZE = 32

    @classmethod
    def decode(cls, buf, offset):
        d = struct.unpack_from('<IIfffHHHhI', buf, offset)
        self = cls.__new__(cls)
        self.dx = d[0]; self.dy = d[1]
        self.parx = d[2]; self.pmra = d[3]; self.pmdec = d[4]
        self.magG = d[5] * 0.001 - 1.5
        self.magBP = d[6] * 0.001 - 1.5
        self.magRP = d[7] * 0.001 - 1.5
        self.dra = d[8]; self.flags = d[9]
        return self


class EncodedStarSPData(EncodedStarData):
    __slots__ = ('flux_min', 'flux_mul', 'flux')

    @classmethod
    def decode(cls, buf, offset):
        self = cls.__new__(cls)
        d = struct.unpack_from('<IIfffHHHhIff', buf, offset)
        self.dx = d[0]; self.dy = d[1]
        self.parx = d[2]; self.pmra = d[3]; self.pmdec = d[4]
        self.magG = d[5] * 0.001 - 1.5
        self.magBP = d[6] * 0.001 - 1.5
        self.magRP = d[7] * 0.001 - 1.5
        self.dra = d[8]; self.flags = d[9]
        self.flux_min = d[10]; self.flux_mul = d[11]
        self.flux = np.frombuffer(buf, np.uint8, WAVELENGTH_COUNT, offset + 40)
        return self

    def decode_spectrum(self, normalize=False, photon_flux=False):
        spec = self.flux.astype(np.float64) * self.flux_mul + self.flux_min
        if photon_flux:
            spec *= _WAVELENGTH / 1.602e-19 / 1.239979e-3
        if normalize:
            mx = spec.max()
            if mx > 0:
                spec /= mx
        return spec


def _ang2pix_nest4(ra, dec):
    nside = 4
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    z = np.cos(theta)
    abs_z = np.abs(z)

    eq_mask = abs_z <= 2.0 / 3.0
    np_mask = z > 2.0 / 3.0
    sp_mask = z < -2.0 / 3.0

    face = np.empty(len(ra), dtype=np.int32)
    ix = np.empty(len(ra), dtype=np.int32)
    iy = np.empty(len(ra), dtype=np.int32)

    if np.any(eq_mask):
        _eq = np.flatnonzero(eq_mask)
        temp1 = nside * (0.5 + phi[_eq] * 0.75 / np.pi)
        temp2 = nside * z[_eq] * 0.75
        jp = np.floor(temp1 - temp2).astype(np.int32)
        jm = np.floor(temp1 + temp2).astype(np.int32)
        ifp = jp // nside
        ifm = jm // nside
        face[_eq] = np.where(
            ifp == ifm,
            np.where(ifp < 0, 4, 0),
            np.where(ifp < ifm,
                     np.where(ifp < 0, 2, 6),
                     np.where(ifm < 0, 4, 8)))
        ix[_eq] = jm % nside
        iy[_eq] = (nside - 1) - (jp % nside)

    if np.any(np_mask):
        _np = np.flatnonzero(np_mask)
        tp = phi[_np] % (np.pi / 2.0)
        sigma = np.where(abs_z[_np] < 0.99,
                         2.0 * (1.0 - abs_z[_np]),
                         np.sin(theta[_np]) ** 2 / abs_z[_np])
        x = nside * sigma * 0.5
        ix_np = np.clip(np.floor(x), 0, nside - 1).astype(np.int32)
        iy_np = ix_np.copy()
        hi_z = abs_z[_np] > 0.99
        jp_np = np.empty(len(_np), dtype=np.int32)
        jm_np = np.empty(len(_np), dtype=np.int32)
        jp_np[hi_z] = np.floor(tp[hi_z] * x[hi_z]).astype(np.int32)
        jm_np[hi_z] = np.floor((np.pi / 2.0 - tp[hi_z]) * x[hi_z]).astype(np.int32)
        lo = ~hi_z
        jp_np[lo] = np.floor((tp[lo] / np.pi * 0.5 + 0.5) * nside * sigma[lo]).astype(np.int32)
        jm_np[lo] = np.floor(((1.0 - tp[lo] / np.pi * 0.5 - 0.5) + 0.5) * nside * sigma[lo]).astype(np.int32)
        ntt = np.floor(phi[_np] * 2.0 / np.pi).astype(np.int32)
        face[_np] = np.where(ntt == 0, 0, np.where(ntt == 1, 2, np.where(ntt == 2, 4, 6)))
        ix[_np] = nside - 1 - ix_np
        iy[_np] = iy_np

    if np.any(sp_mask):
        _sp = np.flatnonzero(sp_mask)
        tp = phi[_sp] % (np.pi / 2.0)
        sigma = np.where(abs_z[_sp] < 0.99,
                         2.0 * (1.0 - abs_z[_sp]),
                         np.sin(theta[_sp]) ** 2 / abs_z[_sp])
        x = nside * sigma * 0.5
        ix_sp = np.clip(np.floor(x), 0, nside - 1).astype(np.int32)
        iy_sp = ix_sp.copy()
        hi_z = abs_z[_sp] > 0.99
        jp_sp = np.empty(len(_sp), dtype=np.int32)
        jm_sp = np.empty(len(_sp), dtype=np.int32)
        jp_sp[hi_z] = np.floor(tp[hi_z] * x[hi_z]).astype(np.int32)
        jm_sp[hi_z] = np.floor((np.pi / 2.0 - tp[hi_z]) * x[hi_z]).astype(np.int32)
        lo = ~hi_z
        jp_sp[lo] = np.floor((tp[lo] / np.pi * 0.5 + 0.5) * nside * sigma[lo]).astype(np.int32)
        jm_sp[lo] = np.floor(((1.0 - tp[lo] / np.pi * 0.5 - 0.5) + 0.5) * nside * sigma[lo]).astype(np.int32)
        ntt = np.floor(phi[_sp] * 2.0 / np.pi).astype(np.int32)
        face[_sp] = np.where(ntt == 0, 8, np.where(ntt == 1, 10, np.where(ntt == 2, 12, 14)))
        ix[_sp] = ix_sp
        iy[_sp] = iy_sp

    pixel = (face % 12) * 16
    scale = 1
    for _ in range(nside.bit_length() - 1):
        pixel += ((ix & 1) + ((iy & 1) << 1)) * scale
        scale <<= 2
        ix >>= 1
        iy >>= 1

    return pixel.astype(np.int32)


class QuadtreeIndexNode:
    __slots__ = ('x0', 'y0', 'x1', 'y1', '_is_leaf',
                 '_block_offset', '_block_size', '_compressed_size',
                 '_child_nw', '_child_ne', '_child_sw', '_child_se')

    def __init__(self, buf, offset):
        self.x0 = struct.unpack_from('<d', buf, offset)[0]
        self.y0 = struct.unpack_from('<d', buf, offset + 8)[0]
        self.x1 = struct.unpack_from('<d', buf, offset + 16)[0]
        self.y1 = struct.unpack_from('<d', buf, offset + 24)[0]

        bo_raw = struct.unpack_from('<Q', buf, offset + 32)[0]
        self._is_leaf = (bo_raw & 0x8000000000000000) != 0
        bs = struct.unpack_from('<I', buf, offset + 40)[0]
        cs = struct.unpack_from('<I', buf, offset + 44)[0]

        if self._is_leaf:
            self._block_offset = bo_raw & 0x7FFFFFFFFFFFFFFF
            self._block_size = bs
            self._compressed_size = cs
            self._child_nw = self._child_ne = self._child_sw = self._child_se = 0
        else:
            self._block_offset = 0
            self._block_size = 0
            self._compressed_size = 0
            self._child_nw = struct.unpack_from('<I', buf, offset + 32)[0]
            self._child_ne = struct.unpack_from('<I', buf, offset + 36)[0]
            self._child_sw = struct.unpack_from('<I', buf, offset + 40)[0]
            self._child_se = struct.unpack_from('<I', buf, offset + 44)[0]

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def block_offset(self):
        return self._block_offset

    @property
    def block_size(self):
        return self._block_size

    @property
    def compressed_size(self):
        return self._compressed_size

    def children(self):
        return [self._child_nw, self._child_ne, self._child_sw, self._child_se]

    def bounding_ra_dec(self, projection, center_ra, center_dec):
        corners = [(self.x0, self.y0), (self.x1, self.y0),
                    (self.x1, self.y1), (self.x0, self.y1)]
        ras, decs = [], []
        for x, y in corners:
            ra, dec = _unproject(x, y, projection, center_ra, center_dec)
            ras.append(ra)
            decs.append(dec)
        return (min(ras), max(ras), min(decs), max(decs))


def _unproject(x, y, projection, center_ra, center_dec):
    if projection == 'Equirectangular':
        return x + center_ra, y
    elif projection == 'AzimuthalEquidistant':
        xr = math.radians(x)
        yr = math.radians(y)
        ra = math.degrees(math.atan2(xr, yr))
        if ra < 0:
            ra += 360
        dec = math.degrees(math.asin(math.cos(math.sqrt(xr*xr + yr*yr))))
        if center_dec < 0:
            dec = -dec
        return ra, dec
    else:
        return x, y


class XPSDFile:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self._f = open(self.filepath, 'rb')
        magic = self._f.read(8)
        if magic != b'XPSD0100':
            raise ValueError(f"Invalid XPSD magic: {magic}")

        header_len = struct.unpack('<I', self._f.read(4))[0]
        self._f.read(4)
        xml_data = self._f.read(header_len).decode('utf-8')
        self._parse_xml(xml_data)

        self._load_quadtrees()
        self._data_position = self._xml_data_position

        self._mm = None
        self._leaf_count = sum(len(t['nodes']) for t in self.trees)

    def _parse_xml(self, xml_data):
        root = ET.fromstring(xml_data)
        ns = {'xpsd': 'http://www.pixinsight.com/xpsd'}

        data_elem = root.find('xpsd:Data', ns)
        if data_elem is not None:
            mags = data_elem.attrib.get('magnitudeRange', '-1.5,26').split(',')
            self.magnitude_low = float(mags[0])
            self.magnitude_high = float(mags[1])
            self._xml_data_position = int(data_elem.attrib.get('position', '0'))
            self.compression = data_elem.attrib.get('compression', '').lower()
            self.parameters = data_elem.attrib.get('parameters', '')

        stats = root.find('xpsd:Statistics', ns)
        self.total_sources = int(stats.attrib.get('totalSources', 0)) if stats is not None else 0

        meta = root.find('xpsd:Metadata', ns)
        self.db_identifier = ''
        if meta is not None:
            for elem in meta:
                if elem.tag == '{http://www.pixinsight.com/xpsd}DatabaseIdentifier':
                    self.db_identifier = (elem.text or '').strip()

        self.trees_config = []
        for tree_elem in root.findall('xpsd:Tree', ns):
            center = tree_elem.attrib.get('center', '0,0').split(',')
            self.trees_config.append({
                'projection': tree_elem.attrib.get('projection', 'Equirectangular'),
                'center_ra': float(center[0]),
                'center_dec': float(center[1]),
                'root_position': int(tree_elem.attrib.get('rootPosition', '0')),
                'node_count': int(tree_elem.attrib.get('nodeCount', '0')),
            })

        self._use_byte_shuffle = '+sh' in self.compression
        self._item_size = 0
        if 'itemSize' in data_elem.attrib if data_elem is not None else False:
            pass
        self.has_spectrum = self.db_identifier == 'GaiaDR3SP'
        self.spectrum_count = WAVELENGTH_COUNT

    def _load_quadtrees(self):
        self.trees = []
        self._tree_data_size = 0
        for cfg in self.trees_config:
            self._f.seek(cfg['root_position'])
            node_count = cfg['node_count']
            node_data = self._f.read(node_count * 48)
            self._tree_data_size += len(node_data)

            nodes = []
            for i in range(node_count):
                nodes.append(QuadtreeIndexNode(node_data, i * 48))

            self.trees.append({'config': cfg, 'nodes': nodes})

    def get_mmap(self):
        if self._mm is None:
            self._f.seek(0, os.SEEK_END)
            file_size = self._f.tell()
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def read_leaf_block(self, block_offset, compressed_size, block_size):
        if self._use_byte_shuffle and self._mm is not None:
            comp = self._mm[self._data_position + block_offset:
                             self._data_position + block_offset + compressed_size]
        else:
            self._f.seek(self._data_position + block_offset)
            comp = self._f.read(compressed_size)

        if compressed_size != block_size:
            data = zlib.decompress(comp)
        else:
            data = comp

        return data

    def decode_leaf(self, node, projection, center_ra, center_dec,
                    mag_low=-1.5, mag_high=26,
                    required_flags=0, inclusion_flags=0, exclusion_flags=0,
                    source_limit=0xFFFFFFFF):
        data = self.read_leaf_block(node.block_offset,
                                     node.compressed_size,
                                     node.block_size)
        n = len(data) // STAR_STRIDE
        if n == 0:
            return []

        stars = []
        for i in range(n):
            sp = EncodedStarSPData.decode(data, i * STAR_STRIDE)
            if required_flags and (sp.flags & required_flags) != required_flags:
                continue
            if inclusion_flags and (sp.flags & inclusion_flags) == 0:
                continue
            if exclusion_flags and (sp.flags & exclusion_flags) != 0:
                continue
            if sp.magG < mag_low or sp.magG > mag_high:
                continue

            x = node.x0 + sp.dx / (3600.0 * 1000.0 * 500.0)
            y = node.y0 + sp.dy / (3600.0 * 1000.0 * 500.0)

            ra, dec = _unproject(x, y, projection, center_ra, center_dec)
            if sp.dra != 0:
                ra += sp.dra / (3600.0 * 1000.0 * 100.0)
            ra %= 360.0

            sp._decoded_ra = ra
            sp._decoded_dec = dec
            stars.append(sp)

            if len(stars) >= source_limit:
                break

        return stars

    def search_quadtree(self, ra, dec, radius_deg, mag_low=-1.5, mag_high=26,
                        required_flags=0, inclusion_flags=0, exclusion_flags=0,
                        source_limit=0xFFFFFFFF):
        """Search using quadtree traversal (PixInsight-compatible)."""
        results = []
        for tree in self.trees:
            cfg = tree['config']
            proj = cfg['projection']
            cra = cfg['center_ra']
            cdec = cfg['center_dec']
            self._search_recursive(tree['nodes'], 0, ra, dec, radius_deg,
                                    proj, cra, cdec, mag_low, mag_high,
                                    required_flags, inclusion_flags, exclusion_flags,
                                    source_limit - len(results), results)
        return results

    def _search_recursive(self, nodes, node_idx, ra, dec, radius_deg,
                          proj, cra, cdec, mag_low, mag_high,
                          req_f, inc_f, exc_f, limit, results):
        if len(results) >= limit:
            return

        node = nodes[node_idx]
        if not self._intersects_node(node, ra, dec, radius_deg, proj, cra, cdec):
            return

        if node.is_leaf:
            stars = self.decode_leaf(node, proj, cra, cdec,
                                      mag_low, mag_high, req_f, inc_f, exc_f,
                                      limit - len(results))
            cos_ra = math.cos(math.radians(ra))
            sin_ra = math.sin(math.radians(ra))
            cos_dec0 = math.cos(math.radians(dec))
            sin_dec0 = math.sin(math.radians(dec))
            cos_r = math.cos(math.radians(radius_deg))
            for s in stars:
                cos_d = cos_dec0 * math.cos(math.radians(s._decoded_dec))
                d_ang = cos_d * (cos_ra * math.cos(math.radians(s._decoded_ra)) +
                                 sin_ra * math.sin(math.radians(s._decoded_ra))) + \
                        sin_dec0 * math.sin(math.radians(s._decoded_dec))
                if d_ang > 1: d_ang = 1
                if d_ang < -1: d_ang = -1
                if math.acos(d_ang) <= math.radians(radius_deg):
                    results.append(s)
                if len(results) >= limit:
                    return
        else:
            for child_idx in node.children():
                if child_idx != 0:
                    self._search_recursive(nodes, child_idx, ra, dec, radius_deg,
                                           proj, cra, cdec, mag_low, mag_high,
                                           req_f, inc_f, exc_f, limit, results)

    def _intersects_node(self, node, ra, dec, radius_deg, proj, cra, cdec):
        ra_min, ra_max, dec_min, dec_max = node.bounding_ra_dec(proj, cra, cdec)

        d_ra = max(0, ra_min - ra, ra - ra_max)
        if d_ra > 180:
            d_ra = 360 - d_ra

        d_dec = max(0, dec_min - dec, dec - dec_max)
        if d_ra < 1e-12 and d_dec < 1e-12:
            return True

        d_deg = math.sqrt(d_ra**2 + d_dec**2)
        return d_deg <= radius_deg * 1.5

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self):
        self.close()
