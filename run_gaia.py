"""
Gaia DR3/SP XPSD Direct Client — 2.19亿星 · 毫秒查询 · 零转换

用法:
  python run_gaia.py --build-index        # 构建外挂索引（首次，6秒）
  python run_gaia.py --benchmark           # 性能基准测试
  python run_gaia.py --ra 180 --dec 0 --radius 5  # 锥形搜索
  python run_gaia.py --source-id 1000000   # 按ID调取光谱
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / 'src'))

from gaia_spectra_store import GaiaClient

DATA_DIR = ROOT / 'data'
INDEX_DIR = ROOT / 'indices'


def main():
    import argparse
    p = argparse.ArgumentParser(description='Gaia DR3/SP XPSD Client')
    p.add_argument('--data-dir', default=str(DATA_DIR))
    p.add_argument('--index-dir', default=str(INDEX_DIR))
    p.add_argument('--build-index', action='store_true', help='构建索引')
    p.add_argument('--ra', type=float, default=180)
    p.add_argument('--dec', type=float, default=0)
    p.add_argument('--radius', type=float, default=5)
    p.add_argument('--source-id', type=int, default=None)
    p.add_argument('--limit', type=int, default=None, help='最大返回数，默认不限')
    p.add_argument('--benchmark', action='store_true')
    args = p.parse_args()

    if args.build_index:
        print('构建外挂索引...'); t0 = time.time()
        client = GaiaClient(args.data_dir, args.index_dir)
        n = client.build_index()
        print(f'完成: {n:,} 叶节点 ({time.time()-t0:.1f}s)')
        return

    print('加载数据库...', end=' ', flush=True); t0 = time.time()
    client = GaiaClient(args.data_dir, args.index_dir)
    s = client.stats
    print(f'{s["files"]}文件 {s["sources"]:,}星 索引:{"✓" if s["indexed"] else "✗"} ({time.time()-t0:.1f}s)')
    print()

    if args.benchmark:
        tests = [(180, 0, 5), (45, 30, 3), (0, -60, 10), (270, 45, 2)]
        for ra, dec, r in tests:
            t0 = time.time()
            stars = client.cone_search(ra, dec, r, limit=1000)
            t = time.time() - t0
            print(f'  search({ra},{dec},{r}°): {len(stars)} stars in {t*1000:.0f}ms')

        print(); t0 = time.time()
        all_stars = client.cone_search(180, 0, 5)
        t = time.time() - t0
        print(f'  全量搜索 (180,0,5°): {len(all_stars):,} stars in {t*1000:.0f}ms')

        t0 = time.time()
        spec = client.get_spectrum(1000000)
        t = time.time() - t0
        if spec:
            print(f'  get_spectrum(1000000): G={spec["phot_g_mean_mag"]:.3f} '
                  f'BP={spec["phot_bp_mean_mag"]:.3f} ({t*1000:.0f}ms)')
        return

    if args.source_id is not None:
        t0 = time.time()
        spec = client.get_spectrum(args.source_id)
        t = time.time() - t0
        if spec:
            print(f'source_id={args.source_id}: ra={spec["ra"]:.6f} dec={spec["dec"]:.6f}')
            print(f'  G={spec["phot_g_mean_mag"]:.3f}  BP={spec["phot_bp_mean_mag"]:.3f}  '
                  f'RP={spec["phot_rp_mean_mag"]:.3f}')
            bp = spec['bp_spectrum']
            print(f'  spectrum: {len(bp)} points [{bp.min():.3e}, {bp.max():.3e}]')
            print(f'  wavelength: {spec["wavelength"][0]:.0f}-{spec["wavelength"][-1]:.0f}nm')
            print(f'  query time: {t*1000:.0f}ms')
        else:
            print(f'source_id={args.source_id}: not found')
        return

    t0 = time.time()
    stars = client.cone_search(args.ra, args.dec, args.radius, limit=args.limit)
    t = time.time() - t0
    print(f'搜索 (ra={args.ra}, dec={args.dec}, r={args.radius}°): '
          f'{len(stars)} 星 ({t*1000:.0f}ms)')
    for i, s in enumerate(stars[:10]):
        print(f'  [{i}] ra={s._decoded_ra:.6f} dec={s._decoded_dec:.6f} '
              f'G={s.magG:.3f} BP={s.magBP:.3f} flags=0x{s.flags:08x}')
    client.close()


if __name__ == '__main__':
    main()
