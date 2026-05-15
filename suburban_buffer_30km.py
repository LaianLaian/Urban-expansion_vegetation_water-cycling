"""
Peri-urban ring: a 30 km annulus outside each urban patch (buffer band).

Geometry is buffer_30km(city) \\ city — the outer 30 km ring excluding the built-up interior
(buffer along the patch outline, not a circle around the centroid).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "filtered_outputs"
FILTERED_2000 = OUTPUT_DIR / "GUB_Global_2000_filtered.shp"
FILTERED_2020 = OUTPUT_DIR / "GUB_Global_2020_filtered.shp"
OUT_SUBURB_2000 = OUTPUT_DIR / "GUB_suburban_30km_2000.shp"
OUT_SUBURB_2020 = OUTPUT_DIR / "GUB_suburban_30km_2020.shp"

BUFFER_M = 30_000.0
BUFFER_CRS = "EPSG:6933"


def remove_shapefile_family(target: Path) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"]:
        candidate = target.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def _resolve_city_id(gdf: gpd.GeoDataFrame) -> pd.Series:
    if "city2000_id" in gdf.columns:
        return gdf["city2000_id"].astype("int64")
    if "city2000_i" in gdf.columns:
        return gdf["city2000_i"].astype("int64")
    raise ValueError("City ID column not found (expected city2000_id or truncated city2000_i).")


def _resolve_orig_fid(gdf: gpd.GeoDataFrame) -> pd.Series:
    for c in gdf.columns:
        if c.upper() == "ORIG_FID":
            return gdf[c]
    raise ValueError("ORIG_FID column not found.")


def _as_single_surface(geom) -> Polygon | MultiPolygon:
    """Collapse holes / MultiPolygon to a single surface for buffer minus urban (peri ring)."""
    if geom is None or geom.is_empty:
        return geom
    g = geom
    if g.geom_type == "Polygon":
        return Polygon(g.exterior.coords)
    if g.geom_type == "MultiPolygon":
        polys = [Polygon(p.exterior.coords) for p in g.geoms if not p.is_empty]
        if not polys:
            return g
        u = unary_union(polys)
        if u.geom_type == "Polygon":
            return u
        if u.geom_type == "MultiPolygon":
            return u
    return g


def suburban_ring(geom, buffer_m: float) -> Polygon | MultiPolygon:
    """30 km annulus: outer edge = patch.buffer(m) minus the patch (excludes built-up interior)."""
    u = _as_single_surface(geom)
    if u is None or u.is_empty:
        return u
    buf = u.buffer(buffer_m)
    ring = buf.difference(u)
    if ring.is_empty:
        return ring
    return ring


def build_suburban_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g["ct2000_id"] = _resolve_city_id(g)
    g["origfid"] = _resolve_orig_fid(g)

    g_m = g.to_crs(BUFFER_CRS)
    rings = [suburban_ring(geom, BUFFER_M) for geom in g_m.geometry]
    g_m = g_m.drop(columns="geometry")
    g_m["geometry"] = rings
    g_m = gpd.GeoDataFrame(g_m, geometry="geometry", crs=BUFFER_CRS)
    g_m["peri_km2"] = g_m.geometry.area / 1_000_000.0

    out = g_m[g_m.geometry.notna() & ~g_m.geometry.is_empty].copy()
    out = out.to_crs("EPSG:4326")
    cols = ["ct2000_id", "origfid", "peri_km2", "geometry"]
    return out[cols]


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not FILTERED_2000.is_file() or not FILTERED_2020.is_file():
        raise SystemExit(
            f"Run filter_gub_cities.py first to produce filtered footprints. Expected:\n  {FILTERED_2000}\n  {FILTERED_2020}"
        )

    print("Reading filtered city footprints...")
    g00 = gpd.read_file(FILTERED_2000)
    g20 = gpd.read_file(FILTERED_2020)

    ids00 = set(_resolve_city_id(g00))
    ids20 = set(_resolve_city_id(g20))
    if ids00 != ids20:
        only0 = ids00 - ids20
        only2 = ids20 - ids00
        raise SystemExit(
            "City ID sets in filtered 2000 vs 2020 shapefiles do not match."
            f" Only in 2000: {len(only0)}; only in 2020: {len(only2)}"
        )

    print(
        f"Building 30 km suburban ring (buffer minus city interior, {BUFFER_CRS}, {BUFFER_M/1000:.0f} km)..."
    )
    sub00 = build_suburban_gdf(g00)
    sub20 = build_suburban_gdf(g20)
    if len(sub00) != len(g00) or len(sub20) != len(g20):
        print(
            f"Note: some cities have empty suburban geometries — 2000: {len(g00)}→{len(sub00)}, "
            f"2020: {len(g20)}→{len(sub20)}"
        )

    common = set(sub00["ct2000_id"]) & set(sub20["ct2000_id"])
    if len(common) < len(sub00) or len(common) < len(sub20):
        print(
            f"Note: writing only cities with valid suburban geometry in both years ({len(common)} cities, keyed by ct2000_id)."
        )
    sub00 = sub00[sub00["ct2000_id"].isin(common)].sort_values("ct2000_id", kind="mergesort")
    sub20 = sub20[sub20["ct2000_id"].isin(common)].sort_values("ct2000_id", kind="mergesort")

    print("Writing shapefiles...")
    remove_shapefile_family(OUT_SUBURB_2000)
    remove_shapefile_family(OUT_SUBURB_2020)
    sub00.to_file(OUT_SUBURB_2000, driver="ESRI Shapefile")
    sub20.to_file(OUT_SUBURB_2020, driver="ESRI Shapefile")

    print(f"Cities (aligned): {len(sub00)}")
    print(f"Year-2000 suburban ring (from 2000 footprint): {OUT_SUBURB_2000}")
    print(f"Year-2020 suburban ring (from 2020 footprint): {OUT_SUBURB_2020}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
