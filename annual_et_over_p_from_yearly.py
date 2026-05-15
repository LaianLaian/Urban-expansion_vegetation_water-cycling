"""
Annual NetCDF grids -> per-cell annual ET/P and ET_vegenon/P (two NetCDF outputs).

Defaults (override with CLI):
  ET_yr.nc, ETyr_vegenon.nc, MSWEP_annual_precip_2000_2021.nc

Alignment: precipitation longitudes wrapped to 0–360°, latitudes sorted ascending,
then nearest-neighbour interpolation onto the ET grid.

Some ET yearly files use non-standard CF time (e.g. ``since 200101``); xarray may fail
to decode times. Use ``decode_times=False`` and coerce ``time`` to integer years (default:
2000, 2001, …).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def assign_sequential_calendar_years(ds: xr.Dataset, start_year: int) -> xr.Dataset:
    """Set ``time`` to start_year, start_year+1, … keeping the same length as before."""
    n = ds.sizes["time"]
    years = np.arange(start_year, start_year + n, dtype=np.int32)
    return ds.assign_coords(
        time=("time", years, {"long_name": "calendar year", "units": "year"})
    )


def open_et_yearly(path: Path, start_year: int) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=False)
    return assign_sequential_calendar_years(ds, start_year)


def open_precip_yearly(path: Path, start_year: int) -> xr.Dataset:
    """Try normal time decoding; on failure open with decode_times=False and assign years."""
    try:
        ds = xr.open_dataset(path)
    except ValueError as e:
        msg = str(e).lower()
        if "decode" in msg or "time" in msg or "failed to decode" in msg:
            ds = xr.open_dataset(path, decode_times=False)
            return assign_sequential_calendar_years(ds, start_year)
        raise
    return normalize_precip_time_to_years(ds, start_year)


def normalize_precip_time_to_years(ds: xr.Dataset, start_year: int) -> xr.Dataset:
    """Precipitation: datetime -> integer year; numeric >=1900 -> int32; else assign by length."""
    n = ds.sizes["time"]
    t = ds["time"]
    vals = t.values
    # numpy datetime64
    if np.issubdtype(vals.dtype, np.datetime64):
        years = t.dt.year.values.astype(np.int32)
        return ds.assign_coords(
            time=("time", years, {"long_name": "calendar year", "units": "year"})
        )
    # Numeric values that look like calendar years
    v = np.asarray(vals, dtype=np.float64)
    if v.size and np.isfinite(v).all() and np.nanmin(v) >= 1900:
        return ds.assign_coords(
            time=("time", v.astype(np.int32), {"long_name": "calendar year", "units": "year"})
        )
    return assign_sequential_calendar_years(ds, start_year)


def _guess_et_var(ds: xr.Dataset) -> str:
    for name in ("ET", "et", "Evap", "evap"):
        if name in ds.data_vars:
            return name
    raise ValueError(f"No ET variable found; data_vars: {list(ds.data_vars)}")


def _guess_p_var(ds: xr.Dataset) -> str:
    for name in ("annual_pricip", "annual_precip", "P", "precipitation", "precip", "pr"):
        if name in ds.data_vars:
            return name
    raise ValueError(f"No precipitation variable found; data_vars: {list(ds.data_vars)}")


def _to_time_lat_lon(da: xr.Dataset | xr.DataArray, var: str) -> xr.DataArray:
    if isinstance(da, xr.Dataset):
        da = da[var]
    dims = set(da.dims)
    if dims == {"time", "lat", "lon"}:
        return da.transpose("time", "lat", "lon")
    if dims == {"lon", "lat", "time"}:
        return da.transpose("time", "lat", "lon")
    if dims == {"lat", "lon", "time"}:
        return da.transpose("time", "lat", "lon")
    raise ValueError(f"Variable {var} dims are not time/lat/lon: {da.dims}")


def prepare_precip_on_et_grid(p_da: xr.DataArray, et_lat: xr.DataArray, et_lon: xr.DataArray) -> xr.DataArray:
    p = p_da.assign_coords(lon=((p_da["lon"] + 360) % 360))
    p = p.sortby("lon").sortby("lat")
    return p.interp(lat=et_lat, lon=et_lon, method="nearest")


def ratio_et_over_p(et: xr.DataArray, p_on_et: xr.DataArray, out_name: str, long_name: str) -> xr.DataArray:
    r = xr.where(p_on_et != 0, et / p_on_et, np.nan)
    r = r.transpose("time", "lat", "lon")
    r.name = out_name
    r.attrs["long_name"] = long_name
    r.attrs["units"] = "1"
    r.coords["time"].attrs.setdefault("long_name", "year")
    r.coords["time"].attrs.setdefault("units", "year")
    return r


def align_et_to_reference(et: xr.DataArray, ref_lat: xr.DataArray, ref_lon: xr.DataArray) -> xr.DataArray:
    if et.sizes["lat"] == ref_lat.size and et.sizes["lon"] == ref_lon.size:
        if np.allclose(et["lat"].values, ref_lat.values) and np.allclose(et["lon"].values, ref_lon.values):
            return et
    return et.interp(lat=ref_lat, lon=ref_lon, method="nearest")


def write_ratio(ds_out: xr.Dataset, path: Path, var_name: str) -> None:
    ds_out.to_netcdf(
        path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan}},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Annual ET and P -> ET/P and ET_vegenon/P NetCDFs")
    ap.add_argument("--et", type=Path, default=PROJECT_ROOT / "ET_yr.nc")
    ap.add_argument(
        "--et-vegenon", type=Path, dest="et_vegenon", default=PROJECT_ROOT / "ETyr_vegenon.nc"
    )
    ap.add_argument(
        "--precip",
        type=Path,
        default=PROJECT_ROOT / "MSWEP_annual_precip_2000_2021.nc",
    )
    ap.add_argument(
        "--out-et-p", type=Path, default=PROJECT_ROOT / "ET_over_P_annual_2000_2021.nc"
    )
    ap.add_argument(
        "--out-et-vegenon-p",
        type=Path,
        default=PROJECT_ROOT / "ET_vegenon_over_P_annual_2000_2021.nc",
    )
    ap.add_argument(
        "--time-start-year",
        type=int,
        default=2000,
        help="When time cannot be decoded, assign consecutive years from this value (ET and P).",
    )
    args = ap.parse_args()

    ds_et = open_et_yearly(args.et, args.time_start_year)
    ds_veg = open_et_yearly(args.et_vegenon, args.time_start_year)
    ds_p = open_precip_yearly(args.precip, args.time_start_year)

    et_v = _guess_et_var(ds_et)
    veg_v = _guess_et_var(ds_veg)
    p_v = _guess_p_var(ds_p)

    et = _to_time_lat_lon(ds_et, et_v)
    et_veg = _to_time_lat_lon(ds_veg, veg_v)
    p = _to_time_lat_lon(ds_p, p_v)

    if et.sizes.get("time") != p.sizes.get("time"):
        raise ValueError(
            f"ET and P time lengths differ: {et.sizes.get('time')} vs {p.sizes.get('time')}"
        )
    if not np.array_equal(np.asarray(et["time"].values), np.asarray(p["time"].values)):
        raise ValueError("ET and P time coordinates do not match; check both files.")

    if et.sizes.get("time") != et_veg.sizes.get("time"):
        raise ValueError(
            f"ET and ET_vegenon time lengths differ: {et.sizes.get('time')} vs {et_veg.sizes.get('time')}"
        )
    if not np.array_equal(np.asarray(et["time"].values), np.asarray(et_veg["time"].values)):
        raise ValueError("ET and ET_vegenon time coordinates do not match; check both files.")

    et_veg = align_et_to_reference(et_veg, et["lat"], et["lon"])

    p_on_et = prepare_precip_on_et_grid(p, et["lat"], et["lon"])

    r1 = ratio_et_over_p(
        et,
        p_on_et,
        out_name="ET_over_P",
        long_name="annual ET / annual precipitation",
    )
    r2 = ratio_et_over_p(
        et_veg,
        p_on_et,
        out_name="ET_vegenon_over_P",
        long_name="annual ET (fixed vegetation) / annual precipitation",
    )

    out1 = r1.to_dataset()
    out1.attrs["note"] = (
        "P aligned: lon 0-360, lat ascending, nearest-neighbor to ET grid; "
        f"ET from {args.et.name}; P from {args.precip.name}."
    )
    write_ratio(out1, args.out_et_p, "ET_over_P")

    out2 = r2.to_dataset()
    out2.attrs["note"] = (
        "P aligned: lon 0-360, lat ascending, nearest-neighbor to ET grid; "
        f"ET_vegenon from {args.et_vegenon.name}; P from {args.precip.name}."
    )
    write_ratio(out2, args.out_et_vegenon_p, "ET_vegenon_over_P")

    ds_et.close()
    ds_veg.close()
    ds_p.close()

    print("Written:", args.out_et_p)
    print("Written:", args.out_et_vegenon_p)


if __name__ == "__main__":
    main()
