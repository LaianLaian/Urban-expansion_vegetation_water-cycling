# Urban expansion reshapes vegetation-driven urban water cycling worldwide

This repository contains the **processing scripts** used to build city- and sub-regional panels for the study: gridded ratios **ET/P**, **ET<sub>fixed‑veg</sub>/P**, optional components (ET, precipitation), and **LAI**, aligned to **Global Urban Boundaries (GUB)**-based **core**, **expansion**, and **peri-urban ring** zones.

> **Note.** Input rasters and shapefiles are not distributed here. Obtain GUB, forcing data (e.g. MSWEP precipitation, your ET products, LAI cubes), and configure paths via CLI arguments or by placing files under the **project root** (the parent directory of this `code/` folder), matching the default filenames below.

---

## Ethics, data use, and reproducibility

- Cite the original sources for **GUB**, **LandScan**, **MSWEP**, **LAI**, and any **ET** products in the manuscript and supplementary information.
- Scripts write **provenance** metadata (`RUN_METADATA*.txt`) listing paths and key parameters used in each run.
- For *Nature* family journals, register a **DOI** for the exact code snapshot (e.g. Zenodo archived release) and pin **Python and package versions** in supplementary material (`pip freeze` / conda export).

---

## Environment

- **Python** 3.10+ recommended  
- **Install dependencies** (from this directory):

```bash
pip install -r requirements.txt
```

Core stack: `numpy`, `pandas`, `xarray`, `geopandas`, `shapely`, `rasterio`, `pyproj`, `scipy`, `netCDF4`; `openpyxl` for Excel output in the LAI workflow; `tqdm` optional for progress bars.

---

## Project layout (typical)

Place or symlink data next to this `code/` folder (i.e. **`code/../` = project root**). Defaults in the scripts assume that layout.

| Role | Typical path (under project root) |
|------|-----------------------------------|
| Raw GUB 2000 / 2020 footprints | `GUB_Global_2000.shp`, `GUB_Global_2020.shp` |
| LandScan 2000 population raster | `landscan-global-2000-assets/landscan-global-2000.tif` |
| Filtered city footprints | `filtered_outputs/GUB_Global_*_filtered.shp` |
| Suburban buffers | `filtered_outputs/GUB_suburban_30km_*.shp` |
| Annual ET/P NetCDF | `ET_over_P_annual_2000_2021.nc`, `ET_vegenon_over_P_annual_2000_2021.nc` |
| City-scale outputs | e.g. `city_etp_outputs/`, `city_lai_outputs/` |

All scripts resolve **`PROJECT_ROOT = parent of this package's directory`** as `Path(__file__).resolve().parent.parent` so that moving the project folder does not hard-code drive letters.

---

## Methodological summary

1. **City sample** (`filter_gub_cities.py`): filter 2000 GUB polygons by footprint area / zonal population; match 2020 footprints to the same cities; export coherent 2000/2020 pairs and a summary CSV.
2. **Peri-urban ring** (`suburban_buffer_30km.py`): 30 km annulus around each footprint (buffer minus urban polygon) in equal-area CRS, then WGS84.
3. **Zonal regions** (`aggregate_etp_to_city_scale.py` and dependents):  
   - **Core** = urban 2000 ∩ urban 2020  
   - **Expansion** = year-2020 footprint minus year-2000 footprint (mutually exclusive with peri after cleaning)  
   - **Peri** = suburban ring minus overlaps with expansion/urban  
4. **Gridded aggregation**: for each polygon, cells are retained if **intersection area / cell area** in cylindrical equal-area projection meets a threshold (`--area-threshold`); optional softer fallback (`--mask-fallback-threshold`). Means use **cos(lat)**-weighted averages over selected cells.
5. **Interpretation**: **V = W1 − W2** with **W1 = ET/P** (changing vegetation scenario) and **W2 = ET<sub>veg2000</sub>/P** (vegetation fixed at 2000). This encodes scenario contrast, **not** a full causal decomposition. Urban effects are described via spatial structure (**core/expansion/peri**) and metrics such as **Expansion** (expansion area / 2020 city area).

---

## Pipeline order

Run in this order unless you already have intermediate products.

### 1. Filter global cities (`filter_gub_cities.py`)

**Inputs:** GUB 2000/2020 shapefiles, LandScan 2000 GeoTIFF.  
**Outputs:** `filtered_outputs/GUB_Global_2000_filtered.shp`, `GUB_Global_2020_filtered.shp`, `filtered_city_summary.csv`.

Edit `WORKSPACE` if needed, or refactor to `argparse` for your archive.

```bash
python filter_gub_cities.py
```

### 2. Build 30 km suburban rings (`suburban_buffer_30km.py`)

**Requires** filtered shapefiles from step 1.  
**Outputs:** `GUB_suburban_30km_2000.shp`, `GUB_suburban_30km_2020.shp` (2020 ring is used as **peri** in later steps; align filenames with aggregation CLI).

```bash
python suburban_buffer_30km.py
```

### 3. (Optional) Monthly → annual ET and P; annual ET/P

- **`build_annual_et_p_standalone.py`**: sum 12 monthly layers to annual ET; annual precipitation; write annual NetCDFs and one **ET/P** file.  
- **`annual_et_over_p_from_yearly.py`**: if you already have **annual** ET and P on compatible time axes, build **ET_over_P** and **ET_vegenon_over_P** NetCDFs with nearest-neighbor P regridding onto the ET grid.

```bash
python build_annual_et_p_standalone.py --out-dir /path/to/project_root
python annual_et_over_p_from_yearly.py --et /path/to/ET_yr.nc --precip /path/to/P_yr.nc ...
```

### 4. Aggregate ET/P (and optionally ET, P) to cities (`aggregate_etp_to_city_scale.py`)

Main panel: **`panel_city_year_region.csv`** (`city_id`, `year`, `region`, `W1`, `W2`, `V`, plus optional `ET`, `ET_vegenon`, `P`).  
City-union summaries, climatologies, attributes, per-city extracts under **`by_city/`**. See script docstring for full output list.

```bash
python aggregate_etp_to_city_scale.py --help
python aggregate_etp_to_city_scale.py --w1 ... --w2 ... --shp-dir .../filtered_outputs --out-dir .../city_etp_outputs
```

### 5. Precipitation-only rerun (`aggregate_p_only_to_city_scale.py`)

Rewrites **P-only** artefacts consistent with step 4 without touching the main panel files; useful after updating precipitation NetCDF while keeping ET/P grids unchanged.

```bash
python aggregate_p_only_to_city_scale.py --out-dir .../city_etp_outputs
```

### 6. LAI aggregation (`aggregate_lai_yearly_to_city_scale.py`)

Stacks monthly LAI NetCDFs (filenames ending in `_YYYY.nc` set calendar year), annual means, same zonal geometry as ET/P pipeline, Excel + CSV trend summaries.

```bash
pip install openpyxl
python aggregate_lai_yearly_to_city_scale.py --lai-dir .../LAI --out-dir .../city_lai_outputs
```

---

## Output artefacts (overview)

| File / directory | Contents |
|------------------|-----------|
| `panel_city_year_region.csv` | Long panel by city, year, region (core/expansion/peri) |
| `city_year_overall.csv` | Union of three regions per city-year |
| `city_attributes.csv` | City-level summaries: means, Theil–Sen trend of V, Expansion ratio, early/late windows |
| `city_region_climatology_*_mean.csv` | Period averages by city and region |
| `RUN_METADATA.txt` | Run parameters and input paths |
| `by_city/<id>_*.csv` | One CSV set per municipality |

*(LAI and P-only runs add analogous filenames — see respective scripts.)*

---

## Performance and memory

- Global high-resolution grids (e.g. 0.05°) multiplied by decades of layers are **memory-intensive**. Use `--interp-res-deg` only when needed; coarse grids combined with `--area-threshold` can yield NaNs for small footprints — then adjust threshold or fallback (document choices in supplementary methods).

---

## Figure source data

Source data used to generate the main-text figures are provided as Excel files:

| File             | Description                 |
| ---------------- | --------------------------- |
| `Sourcedata_Fig2.xlsx` | Source data used for Fig. 2 |
| `Sourcedata_Fig3.xlsx` | Source data used for Fig. 3 |
| `Sourcedata_Fig4.xlsx` | Source data used for Fig. 4 |
| `Sourcedata_Fig5.xlsx` | Source data used for Fig. 5 |

Figure 1 is a conceptual framework and therefore has no associated source data file.

---

## Citation

If you use this code, cite the accompanying paper once published and the third-party datasets (GUB, MSWEP, LandScan, etc.) per their licenses.

---

