# Gaia DR3/SP XPSD Data Files

## 数据下载

XPSD 数据库文件来自 **PixInsight Gaia DR3/SP XPSD Database v1.0.0**。

请从 PixInsight 官方下载以下 20 个文件并放置到此目录：

```
gdr3sp-1.0.0-01.xpsd  (4.0 GB)
gdr3sp-1.0.0-02.xpsd  (4.0 GB)
gdr3sp-1.0.0-03.xpsd  (4.0 GB)
...
gdr3sp-1.0.0-20.xpsd  (4.0 GB)
```

### 下载方式

1. **PixInsight 官方**（推荐）：
   - 安装 PixInsight → Resources → Updates → Manage Repositories
   - 添加 Gaia DR3/SP 数据库 → 下载

2. **手动下载**：
   访问 PixInsight 分发服务器：
   ```
   https://dist.pixinsight.com/
   ```

### 数据说明

| 属性 | 值 |
|------|-----|
| 恒星数量 | 219,165,266 |
| 波长范围 | 336–1020 nm |
| 光谱采样 | 343 点（步长 2 nm） |
| 星等范围 | -2.0 ~ +13.62 |
| 总大小 | ~80 GB |

### 首次运行

下载完成后，构建外挂索引（仅需 6 秒，生成 ~154 MB）：

```bash
python run_gaia.py --build-index
```

---

## Attribution

This product uses data from the European Space Agency (ESA) mission Gaia
(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and
Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium).

XPSD format and database © Pleiades Astrophoto S.L. (PixInsight)
