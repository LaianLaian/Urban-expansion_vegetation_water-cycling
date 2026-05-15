"""
Monthly ET / ET_vegenon / precipitation → annual totals (sum 12 months) and annual ET/P.

Assumptions:
- Time dimension spans 264 months (2000-01 … 2021-12)
- ET datasets use variable name ``ET`` and dimensions ``(lon, lat, time)``; recommended ``decode_times=False``
- Precipitation uses variable name ``precipitation`` with dimensions ``(time, lat, lon)``

Note: in some environments xarray lazy-loading of ET_monvegenon.nc can raise Permission denied;
annual ET is therefore aggregated with ``netCDF4`` in monthly chunks for robustness and memory use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_MONTHS = 264
N_YEARS = N_MONTHS // 12
YEARS = np.arange(2000, 2000 + N_YEARS, dtype=np.int32)


def annual_et_netcdf4(in_path: Path, out_path: Path) -> xr.Dataset:
    """Read ET(lon,lat,time); write Dataset ET(time,lat,lon), time years 2000..2021."""
    ds_nc = Dataset(str(in_path), "r")
    try:
        lon = np.asarray(ds_nc.variables["lon"][:], dtype=np.float32)
        lat = np.asarray(ds_nc.variables["lat"][:], dtype=np.float32)
        v = ds_nc.variables["ET"]
        nlon, nlat, nt = v.shape
        if nt != N_MONTHS:
            raise ValueError(f"{in_path.name}: expected time length {N_MONTHS}, got {nt}")

        annual = np.empty((N_YEARS, nlat, nlon), dtype=np.float32)
        acc = np.empty((nlon, nlat), dtype=np.float64)
        for yi in range(N_YEARS):
            t0 = yi * 12
            acc.fill(0.0)
            for k in range(12):
                acc += v[:, :, t0 + k]
            annual[yi, :, :] = acc.T.astype(np.float32, copy=False)

        out = xr.Dataset(
            data_vars={
                "ET": (
                    ("time", "lat", "lon"),
                    annual,
                    {
                        "long_name": "ET annual sum",
                        "units": "ET-mon per year",
                    },
                )
            },
            coords={
                "time": ("time", YEARS, {"long_name": "year", "units": "year"}),
                "lat": ("lat", lat, {"long_name": "latitude"}),
                "lon": ("lon", lon, {"long_name": "longitude"}),
            },
            attrs={"source": str(in_path), "note": "sum of 12 monthly ET values per year"},
        )
    finally:
        ds_nc.close()

    out.to_netcdf(
        out_path,
        encoding={"ET": {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    return out


def prepare_precip_grid(da: xr.DataArray) -> xr.DataArray:
    da = da.assign_coords(lon=((da["lon"] + 360) % 360))
    return da.sortby("lon").sortby("lat")


def annual_precip_xarray(in_path: Path, out_path: Path) -> xr.Dataset:
    ds = xr.open_dataset(in_path)
    if ds.sizes.get("time") != N_MONTHS:
        raise ValueError(
            f"{in_path.name}: expected time length {N_MONTHS}, got {ds.sizes.get('time')}"
        )

    chunks = []
    for yi in range(N_YEARS):
        m = ds["precipitation"].isel(time=slice(yi * 12, (yi + 1) * 12))
        a = m.sum(dim="time", keep_attrs=True).expand_dims(time=[YEARS[yi]])
        chunks.append(a)
    annual = xr.concat(chunks, dim="time")
    annual = prepare_precip_grid(annual).transpose("time", "lat", "lon")
    annual = annual.rename("P")
    annual["P"].attrs = dict(ds["precipitation"].attrs)
    annual["P"].attrs["long_name"] = annual["P"].attrs.get("long_name", "precipitation") + " annual sum"
    if annual["P"].attrs.get("units") == "mm month-1":
        annual["P"].attrs["units"] = "mm yr-1"
    annual["time"].attrs = {"long_name": "year", "units": "year"}
    out = annual.to_dataset()
    out.attrs = dict(ds.attrs)
    out.attrs["history"] = (
        str(ds.attrs.get("history", "")).strip() + " | annual sum of 12 monthly values"
    ).strip(" |")
    out.to_netcdf(
        out_path,
        encoding={"P": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan}},
    )
    ds.close()
    return out


def annual_et_over_p(et_ds: xr.Dataset, p_ds: xr.Dataset, out_path: Path) -> None:
    p_on_et = p_ds["P"].interp(lat=et_ds["lat"], lon=et_ds["lon"], method="nearest")
    ratio = xr.where(p_on_et != 0, et_ds["ET"] / p_on_et, np.nan)
    ratio = ratio.transpose("time", "lat", "lon")
    ratio.name = "ET_over_P"
    ratio.attrs["long_name"] = "annual ET / annual P"
    ratio.attrs["units"] = "1"
    ratio.coords["time"].attrs = {"long_name": "year", "units": "year"}

    out = ratio.to_dataset()
    out.attrs["note"] = (
        "P: lon wrapped to 0-360, lat sorted ascending; "
        "then nearest-neighbor to ET lat/lon for division."
    )
    out.to_netcdf(
        out_path,
        encoding={"ET_over_P": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan}},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--et", type=Path, default=PROJECT_ROOT / "ET_mon.nc")
    ap.add_argument(
        "--et-vegenon", type=Path, dest="et_vegenon", default=PROJECT_ROOT / "ET_monvegenon.nc"
    )
    ap.add_argument(
        "--precip",
        type=Path,
        default=PROJECT_ROOT / "MSWEP_monthly_precip_2000_2021.nc",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    et_out = out_dir / "ET_annual_2000_2021.nc"
    et_veg_out = out_dir / "ET_vegenon_annual_2000_2021.nc"
    p_out = out_dir / "P_annual_2000_2021.nc"
    ratio_out = out_dir / "ET_over_P_annual_2000_2021.nc"

    print("ET annual …", flush=True)
    et_ds = annual_et_netcdf4(args.et, et_out)
    print(et_out, flush=True)

    print("ET_vegenon annual …", flush=True)
    annual_et_netcdf4(args.et_vegenon, et_veg_out)
    print(et_veg_out, flush=True)

    print("P annual …", flush=True)
    p_ds = annual_precip_xarray(args.precip, p_out)
    print(p_out, flush=True)

    print("ET/P …", flush=True)
    annual_et_over_p(et_ds, p_ds, ratio_out)
    print(ratio_out, flush=True)


if __name__ == "__main__":
    main()
