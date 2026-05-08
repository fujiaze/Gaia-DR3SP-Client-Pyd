# Gaia DR3/SP XPSD Direct Client v2.0

**2.19亿恒星 · 毫秒查询 · BP/RP 光谱 · 零数据转换**

直接读取 PixInsight XPSD 格式的 Gaia DR3 恒星光谱数据库，提供高性能锥形搜索和按 ID 光谱调取。核心模块 Cython 编译为 `.pyd`（Windows 原生 DLL）。

---

## 目录结构

```
gaia/
├── data/                          ← 20 个 .xpsd 数据库文件 (~80 GB)
│   ├── gdr3sp-1.0.0-01.xpsd
│   ├── ...
│   └── gdr3sp-1.0.0-20.xpsd
├── indices/                       ← 外挂 HEALPix 索引 (~100 MB，自动生成)
│   └── gdr3sp-1.0.0-*.xpsi
├── src/                           ← Python 源码包
│   ├── setup.py                   ← Cython 编译脚本
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── gaia_spectra_store/
│       ├── __init__.py             ← 公共 API：GaiaClient
│       ├── xpsd_client.py/.pyd     ← XPSD 解析器（Cython DLL，212 KB）
│       ├── _healpix.pyx/.pyd       ← HEALPix 数学（Cython DLL，49 KB）
│       └── index_builder.py        ← 索引构建器
├── run_gaia.py                     ← 一键运行脚本
└── README.md                       ← 本文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install numpy msgpack tqdm cython
```

### 2. 编译 .pyd（首次或源码修改后）

```bash
cd src
python setup.py build_ext --inplace
```

编译产物：
- `_healpix.cp312-win_amd64.pyd` — HEALPix 球面像素化（49 KB）
- `xpsd_client.cp312-win_amd64.pyd` — XPSD 解析器（212 KB）

### 3. 构建外挂索引（首次运行）

```bash
python run_gaia.py --build-index
# 输出: 完成: 685,338 叶节点 (6s)
```

索引仅依赖 `numpy + msgpack`，无需编译。索引文件体积约 100 MB，＜原始数据的 0.2%。

### 4. 运行

```bash
# 锥形搜索
python run_gaia.py --ra 180 --dec 0 --radius 5

# 光谱调取
python run_gaia.py --source-id 1000000

# 性能基准
python run_gaia.py --benchmark
```

---

## API 参考

```python
from gaia_spectra_store import GaiaClient

client = GaiaClient("data/", "indices/")

# 锥形搜索（返回半径内全部恒星）
stars = client.cone_search(ra=180.0, dec=0.0, radius_deg=5.0,
                            mag_low=-1.5, mag_high=26.0,
                            limit=None)   # None = 不限

for s in stars:
    print(s._decoded_ra, s._decoded_dec, s.magG, s.magBP, s.magRP)
    spec = s.decode_spectrum(normalize=False, photon_flux=False)
    # → float64[343] W·m⁻²·nm⁻¹

# 按 ID 调取光谱
spec = client.get_spectrum(source_id=1000000)
# spec['bp_spectrum']      → float64[343]
# spec['rp_spectrum']      → float64[343]  (same as bp for DR3/SP)
# spec['wavelength']       → float64[343]  336-1020 nm
# spec['phot_g_mean_mag']  → float
# spec['phot_bp_mean_mag'] → float
# spec['parallax']         → float (mas)
# spec['pmra'] / spec['pmdec'] → float (mas/yr)
# spec['flags']            → int (GaiaStarFlag)

client.close()
```

### GaiaStarFlag 标志位

| 值 | 含义 |
|-----|------|
| `0x000800F0` | GoldAstrometry（全部黄金标准自测量） |
| `0x00800F00` | SilverAstrometry |
| `0x00070000` | GoldPhotometry（全部黄金标准测光） |
| `0x00000001` | NoPM（无自行/视差） |
| `0x10000000` | BPRPExcess（BP-RP 过量因子 ≥2.0） |

---

## 性能

| 操作 | 延迟 | 说明 |
|------|------|------|
| 加载 20 文件 | 2.6s | 解析 XML header + 加载索引到内存 |
| 锥形搜索 5°（1000星） | 18ms | 跨全部 20 个文件 |
| 锥形搜索 5°（全量） | 51ms | 返回 3,171 星 |
| 锥形搜索 10°（全量） | ~80ms | 返回 ~12,000 星 |
| 单星光谱调取 | 1-2ms | 二分查找 source_id |
| 索引构建 | 6s | 685,338 叶节点 / 20 文件 |

测试环境：Windows 10, Python 3.12, Intel i9, NVMe SSD

---

## 架构设计

### 数据流

```
用户查询 (ra, dec, radius)
  │
  ├─ HEALPix nside=4 粗筛 → 目标像素集
  │     └─ 外挂索引 (.xpsi) 定位叶节点列表
  │
  ├─ 叶节点 AABB 相交测试 → 候选叶节点
  │
  ├─ 读取 XPSD 数据块 (zlib 解压 + byte shuffle)
  │     └─ NumPy strides 向量化解码（无 Python 循环）
  │
  ├─ 星等过滤 (mg/mbp/mrp) + 大圆距离过滤 → 均为向量化
  │
  └─ 返回 EncodedStarSPData 列表
```

### XPSD 数据布局（来自 PCL 源码逆向）

```
文件头（16 bytes）:
  magic[8] = "XPSD0100"
  headerLength[4] (XML header 长度)
  reserved[4]

XML Header: <xpsd><Data/> <Tree/> ×N <Metadata/> <Statistics/></xpsd>

Quadtree 索引节点（48 bytes/节点）:
  x0, y0, x1, y1 (4×double)      ← 投影坐标边界
  blockOffsetAndLeafFlag (uint64) ← 最高位=leaf flag
  blockSize, compressedSize (2×uint32)

EncodedStarSPData（384 bytes/星）:
  dx, dy      uint32   ← 相对节点原点的投影坐标 (0.002 mas)
  parx        float    ← 视差 (mas)
  pmra, pmdec float    ← 自行 (mas/yr)
  magG/BP/RP  uint16   ← (星等+1.5)×1000
  dra         int16    ← 高纬 RA 修正 (0.01 mas)
  flags       uint32   ← 质量标志位
  fluxMin     float    ← 光谱最小值
  fluxMul     float    ← (max-min)/(2^bits-1)
  flux[343]   uint8    ← BP/RP 差分光谱
  [padding 1 byte]     ← 仅当 count 为奇数时
```

### 投影类型

| 类型 | 中心 | 用途 |
|------|------|------|
| Equirectangular | (45°N,0°), (135°N,0°), (225°N,0°), (315°N,0°) | 赤道区域 |
| AzimuthalEquidistant | (0°,90°N), (0°,90°S) | 极区 |

### 解码公式

```
ra  = centerRA + (x0 + dx/1.8e8)       [Equirectangular]
dec = y0 + dy/1.8e8

mag = mag_raw/1000 - 1.5

spectrum[i] = fluxMin + flux_raw[i] * fluxMul   [W·m⁻²·nm⁻¹]
wavelength[i] = 336.0 + i × 2.0   [nm]
photon_flux = spectrum × λ / (h·c)  → ph·s⁻¹·m⁻²·nm⁻¹
```

---

## 数据来源

- **Gaia DR3/SP XPSD Database v1.0.0** — PixInsight 官方发布
- 包含 219,165,266 颗恒星的坐标、星等、自行、视差和 BP/RP 采样光谱
- 波长覆盖：336–1020 nm（343 个采样点，步长 2 nm）
- 星等范围：-2.0 ～ +13.62

### 数据下载

| 方式 | 链接 | 说明 |
|------|------|------|
| 百度网盘 | https://pan.baidu.com/s/1u8CCMtecsaiz2nVjLsThRg?pwd=fujz 提取码 `fujz` | 推荐国内用户，~80 GB |
| PixInsight 官方 | https://dist.pixinsight.com/ | 原始分发服务器 |

详细说明见 [data/README.md](data/README.md)。

## 编译说明

依赖 MSVC Build Tools（Windows）或 GCC（Linux）。

```bash
cd src
python setup.py build_ext --inplace
```

Cython 编译指令：
- `boundscheck=False` — 关闭数组边界检查
- `wraparound=False` — 关闭负索引
- `cdivision=True` — C 级别除法
- `/O2` (MSVC) / `-O3` (GCC) — 最大优化

## 许可证

本项目仅用于学术研究和教育目的。Gaia 数据版权归 ESA/Gaia/DPAC 所有。

PCL (PixInsight Class Library) 头文件版权归 Pleiades Astrophoto S.L. 所有，遵循 PCL 许可证条款。
