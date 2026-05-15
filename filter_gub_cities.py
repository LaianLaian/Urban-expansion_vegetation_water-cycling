from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping


# Project root: parent of this `code/` directory (adjust if you move scripts).
WORKSPACE = Path(__file__).resolve().parent.parent
SHP_2000 = WORKSPACE / "GUB_Global_2000.shp"
SHP_2020 = WORKSPACE / "GUB_Global_2020.shp"
POP_RASTER = WORKSPACE / "landscan-global-2000-assets" / "landscan-global-2000.tif"
OUTPUT_DIR = WORKSPACE / "filtered_outputs"
OUT_2000 = OUTPUT_DIR / "GUB_Global_2000_filtered.shp"
OUT_2020 = OUTPUT_DIR / "GUB_Global_2020_filtered.shp"
OUT_CSV = OUTPUT_DIR / "filtered_city_summary.csv"

AREA_THRESHOLD = 100.0
POP_THRESHOLD = 150000.0
CHUNK_SIZE = 500
MATCH_CRS = "EPSG:6933"


def iter_chunks(total: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def compute_population_sums(gdf: gpd.GeoDataFrame, raster_path: Path, nodata: float) -> pd.Series:
    """Zonal sum via rasterio.mask (all_touched=False); no rasterstats/fiona."""
    sums: list[float] = []
    total = len(gdf)
    with rasterio.open(raster_path) as src:
        g_proj = gdf.to_crs(src.crs)
        for start, end in iter_chunks(total, CHUNK_SIZE):
            chunk = g_proj.iloc[start:end]
            for geom in chunk.geometry:
                if geom is None or geom.is_empty:
                    sums.append(0.0)
                    continue
                try:
                    out_image, _ = mask(
                        src,
                        [mapping(geom)],
                        crop=True,
                        nodata=nodata,
                        all_touched=False,
                    )
                except ValueError:
                    sums.append(0.0)
                    continue
                arr = out_image[0]
                valid = arr != nodata
                sums.append(float(np.sum(arr[valid].astype(np.float64))))
            print(f"Population extraction: {end}/{total}")
    return pd.Series(sums, index=gdf.index, dtype="float64")


def remove_shapefile_family(target: Path) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"]:
        candidate = target.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def match_2020_by_location(
    selected_2000: gpd.GeoDataFrame,
    gdf_2020: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    selected_2000 = selected_2000.copy().reset_index(drop=True)
    selected_2000["city2000_id"] = selected_2000.index.astype("int64")
    gdf_2020 = gdf_2020.copy().reset_index(drop=True)
    gdf_2020["poly2020_id"] = gdf_2020.index.astype("int64")

    left = selected_2000[["city2000_id", "ORIG_FID", "geometry"]]
    right = gdf_2020[["poly2020_id", "urbanArea", "geometry"]]
    joined = gpd.sjoin(left, right, how="inner", predicate="intersects")
    if joined.empty:
        return gdf_2020.iloc[0:0].copy(), pd.DataFrame(columns=["city2000_id", "a20_km2", "part_ct"])

    left_match = left.to_crs(MATCH_CRS).rename_geometry("geometry_2000")
    right_match = right.to_crs(MATCH_CRS).rename_geometry("geometry_2020")
    pairs = (
        joined[["city2000_id", "poly2020_id"]]
        .drop_duplicates()
        .merge(left_match[["city2000_id", "geometry_2000"]], on="city2000_id", how="left")
        .merge(right_match[["poly2020_id", "geometry_2020"]], on="poly2020_id", how="left")
    )
    pairs["ovlp_m2"] = pairs.apply(
        lambda row: row["geometry_2000"].intersection(row["geometry_2020"]).area,
        axis=1,
    )
    pairs = pairs[pairs["ovlp_m2"] > 0].copy()
    if pairs.empty:
        return gdf_2020.iloc[0:0].copy(), pd.DataFrame(columns=["city2000_id", "a20_km2", "part_ct"])

    best = (
        pairs.sort_values(["poly2020_id", "ovlp_m2"], ascending=[True, False])
        .drop_duplicates(subset="poly2020_id", keep="first")
        .copy()
    )
    assigned_2020 = gdf_2020.merge(best[["poly2020_id", "city2000_id"]], on="poly2020_id", how="inner")
    part_counts = assigned_2020.groupby("city2000_id").size().rename("part_ct")
    city_2020 = (
        assigned_2020[["city2000_id", "urbanArea", "geometry"]]
        .dissolve(by="city2000_id", aggfunc={"urbanArea": "sum"})
        .reset_index()
        .merge(part_counts, on="city2000_id", how="left")
    )
    city_2020["a20_km2"] = city_2020["urbanArea"].astype("float64")
    return city_2020, best[["city2000_id", "poly2020_id", "ovlp_m2"]].copy()


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Reading input shapefiles...")
    gdf_2000 = gpd.read_file(SHP_2000)
    gdf_2020 = gpd.read_file(SHP_2020)

    if gdf_2000["ORIG_FID"].duplicated().any():
        raise ValueError("`GUB_Global_2000.shp` contains duplicated `ORIG_FID` values.")

    with rasterio.open(POP_RASTER) as src:
        raster_nodata = float(src.nodata) if src.nodata is not None else -2147483647.0

    print("Computing 2000 city populations from LandScan 2000...")
    gdf_2000 = gdf_2000.copy()
    gdf_2000["pop2000"] = compute_population_sums(gdf_2000, POP_RASTER, raster_nodata)

    print("Applying 2000 area/population filter...")
    gdf_2000["a00_km2"] = gdf_2000["urbanArea"].astype("float64")
    selected_2000 = gdf_2000[
        (gdf_2000["a00_km2"] >= AREA_THRESHOLD) | (gdf_2000["pop2000"] > POP_THRESHOLD)
    ].copy()
    selected_2000 = selected_2000.reset_index(drop=True)
    selected_2000["city2000_id"] = selected_2000.index.astype("int64")

    print("Matching 2020 polygons to 2000 cities by spatial location...")
    gdf_2020_city, matched_pairs = match_2020_by_location(selected_2000, gdf_2020)

    print("Comparing 2000 and 2020 areas...")
    merged = selected_2000.merge(
        gdf_2020_city[["city2000_id", "a20_km2", "part_ct"]],
        on="city2000_id",
        how="left",
    )
    merged["grow_pct"] = ((merged["a20_km2"] - merged["a00_km2"]) / merged["a00_km2"]) * 100.0
    final_2000 = merged[merged["a20_km2"].notna()].copy()

    final_ids = final_2000["city2000_id"].tolist()
    final_2020 = gdf_2020_city[gdf_2020_city["city2000_id"].isin(final_ids)].copy()
    final_2020 = final_2020.merge(
        final_2000[["city2000_id", "ORIG_FID", "pop2000", "a00_km2", "grow_pct"]],
        on="city2000_id",
        how="left",
    )
    final_2020 = final_2020.rename(columns={"urbanArea": "urbanArea20"})
    final_2000 = final_2000.rename(columns={"urbanArea": "urbanArea00"})

    final_2000 = final_2000[
        ["city2000_id", "ORIG_FID", "urbanArea00", "pop2000", "a00_km2", "a20_km2", "grow_pct", "part_ct", "geometry"]
    ].copy()
    final_2020 = final_2020[
        ["city2000_id", "ORIG_FID", "urbanArea20", "pop2000", "a00_km2", "a20_km2", "grow_pct", "part_ct", "geometry"]
    ].copy()

    final_2000["part_ct"] = final_2000["part_ct"].fillna(0).astype("int32")
    final_2020["part_ct"] = final_2020["part_ct"].fillna(0).astype("int32")

    summary = final_2000.drop(columns="geometry").sort_values("grow_pct", ascending=False)

    print("Writing filtered outputs...")
    remove_shapefile_family(OUT_2000)
    remove_shapefile_family(OUT_2020)
    final_2000.to_file(OUT_2000, driver="ESRI Shapefile")
    final_2020.to_file(OUT_2020, driver="ESRI Shapefile")
    summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Selected 2000 cities after area/population filter: {len(selected_2000)}")
    print(f"Matched 2020 polygons by location: {len(matched_pairs)}")
    print(f"Final cities with matched 2020 geometry: {len(final_2000)}")
    print(f"2000 output: {OUT_2000}")
    print(f"2020 output: {OUT_2020}")
    print(f"Summary CSV: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
