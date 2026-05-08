import msgpack
import math
import struct
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .xpsd_client import XPSDFile, QuadtreeIndexNode, _ang2pix_nest4


def build_xpsi_index(xpsd_path, output_path):
    xpsd = XPSDFile(xpsd_path)

    leaves = []
    sid_offset = 0
    for tree in xpsd.trees:
        cfg = tree['config']
        proj = cfg['projection']
        cra = cfg['center_ra']
        cdec = cfg['center_dec']
        for node_idx, node in enumerate(tree['nodes']):
            if node.is_leaf:
                ra_min, ra_max, dec_min, dec_max = node.bounding_ra_dec(proj, cra, cdec)
                from .xpsd_client import STAR_STRIDE
                star_count = node.block_size // STAR_STRIDE
                leaves.append({
                    'tree_idx': xpsd.trees.index(tree),
                    'node_idx': node_idx,
                    'x0': node.x0,
                    'y0': node.y0,
                    'x1': node.x1,
                    'y1': node.y1,
                    'ra_min': ra_min,
                    'ra_max': ra_max,
                    'dec_min': dec_min,
                    'dec_max': dec_max,
                    'block_offset': node.block_offset,
                    'block_size': node.block_size,
                    'compressed_size': node.compressed_size,
                    'projection': proj,
                    'center_ra': cra,
                    'center_dec': cdec,
                    'sid_offset': sid_offset,
                    'star_count': star_count,
                })
                sid_offset += star_count

    if not leaves:
        raise ValueError(f"No leaf nodes found in {xpsd_path}")

    ra_centers = [(l['ra_min'] + l['ra_max']) / 2 for l in leaves]
    dec_centers = [(l['dec_min'] + l['dec_max']) / 2 for l in leaves]

    if len(ra_centers) > 10000:
        ra_centers = np.array(ra_centers, dtype=np.float64)
        dec_centers = np.array(dec_centers, dtype=np.float64)
        hp_pixels = _ang2pix_nest4(ra_centers, dec_centers)
    else:
        hp_pixels = []
        for ra_c, dec_c in zip(ra_centers, dec_centers):
            arr = _ang2pix_nest4(np.array([ra_c]), np.array([dec_c]))
            hp_pixels.append(int(arr[0]))
        hp_pixels = np.array(hp_pixels, dtype=np.int32)

    for i, l in enumerate(leaves):
        l['healpix_4'] = int(hp_pixels[i])

    hp_to_leaves = {}
    for i, l in enumerate(leaves):
        pix = l['healpix_4']
        if pix not in hp_to_leaves:
            hp_to_leaves[pix] = []
        hp_to_leaves[pix].append(i)

    for pix in hp_to_leaves:
        hp_to_leaves[pix].sort(key=lambda i: leaves[i]['ra_min'])

    proj = xpsd.trees[0]['config']['projection']
    center_ra = xpsd.trees[0]['config']['center_ra']
    center_dec = xpsd.trees[0]['config']['center_dec']

    index_data = msgpack.packb({
        'version': 1,
        'projection': proj,
        'center_ra': center_ra,
        'center_dec': center_dec,
        'total_sources': xpsd.total_sources,
        'magnitude_low': xpsd.magnitude_low,
        'magnitude_high': xpsd.magnitude_high,
        'leaf_count': len(leaves),
        'leaf_bbox': [
            {'ra_min': l['ra_min'], 'ra_max': l['ra_max'],
             'dec_min': l['dec_min'], 'dec_max': l['dec_max'],
             'x0': l['x0'], 'y0': l['y0'], 'x1': l['x1'], 'y1': l['y1'],
             'tree_idx': l['tree_idx'], 'node_idx': l['node_idx'],
             'block_offset': l['block_offset'],
             'block_size': l['block_size'],
             'compressed_size': l['compressed_size'],
             'sid_offset': l['sid_offset'],
             'star_count': l['star_count'],
             'healpix_4': l['healpix_4']}
            for l in tqdm(leaves, desc="  打包", leave=False)
        ],
        'hp_index': {str(k): v for k, v in hp_to_leaves.items()},
    })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(index_data)

    xpsd.close()
    return out_path, len(leaves)


def build_all_indices(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xpsd_files = sorted(input_dir.glob('*.xpsd'))
    if not xpsd_files:
        raise ValueError(f"No .xpsd files in {input_dir}")

    print(f"构建索引: {len(xpsd_files)} 个 XPSD 文件")
    total_leaves = 0
    for xp in tqdm(xpsd_files, desc="索引构建", unit="文件"):
        out_path = output_dir / xp.with_suffix('.xpsi').name
        if out_path.exists():
            continue
        _, n = build_xpsi_index(xp, out_path)
        total_leaves += n

    print(f"完成: {total_leaves:,} 个叶节点 → {output_dir}")
    return total_leaves
