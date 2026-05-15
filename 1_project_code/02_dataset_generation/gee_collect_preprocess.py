import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Fix console encoding issue on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import ee
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from dataset_paths import CASES_CONFIG_PATH, resolve_project_path

try:
    import geemap
except Exception:
    geemap = None


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def daterange_daily(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def select_days(start_date: str, end_date: str, max_days=None, tail_days=None):
    days = list(daterange_daily(start_date, end_date))
    if max_days is not None and tail_days is not None:
        raise ValueError("Use only one of --max-days or --tail-days.")
    if max_days is not None:
        return days[:max_days]
    if tail_days is not None:
        return days[-tail_days:]
    return days


def resolve_aoi(case_cfg: dict, config_dir: Path) -> ee.Geometry:
    asset = (case_cfg.get("aoi_asset") or "").strip()
    geojson_path = (case_cfg.get("aoi_geojson") or "").strip()

    if asset and "your_username" not in asset:
        print(f"[AOI] Using asset: {asset}")
        return ee.FeatureCollection(asset).geometry()

    if geojson_path:
        if geemap is None:
            raise RuntimeError("AOI geojson requires geemap, but geemap is unavailable.")
        local_path = Path(geojson_path)
        if not local_path.is_absolute():
            local_path = resolve_project_path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"AOI geojson not found: {local_path}")
        print(f"[AOI] Using local geojson: {local_path}")
        aoi_obj = geemap.geojson_to_ee(str(local_path))
        try:
            return aoi_obj.geometry()
        except Exception:
            return ee.FeatureCollection(aoi_obj).geometry()

    raise ValueError("AOI is not configured. Set aoi_asset or aoi_geojson.")


def export_local(image: ee.Image, out_path: Path, aoi: ee.Geometry, scale: int, crs: str):
    if geemap is None:
        raise RuntimeError("Local export requires geemap.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geemap.ee_export_image(
        image.clip(aoi),
        filename=str(out_path),
        scale=scale,
        region=aoi,
        crs=crs,
        file_per_band=False,
    )


def export_local_with_scale_fallback(
    image: ee.Image,
    out_path: Path,
    aoi: ee.Geometry,
    base_scale: int,
    crs: str,
    scale_candidates: list[int],
):
    last_error = None
    for s in [base_scale] + [x for x in scale_candidates if x != base_scale]:
        try:
            if out_path.exists():
                out_path.unlink()
            export_local(image, out_path, aoi, s, crs)
            if not out_path.exists() or out_path.stat().st_size == 0:
                print(f"[WARN] Export produced no file at scale={s}m for {out_path.name}, retrying...")
                continue
            if s != base_scale:
                print(f"[WARN] Fallback scale used for {out_path.name}: {s}m")
            return
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "Total request size" in msg:
                print(f"[WARN] Request too large at scale={s}m for {out_path.name}, retrying...")
                continue
            raise
    raise RuntimeError(f"Failed export after scale fallback for {out_path}: {last_error}")


# ---------------------------------------------------------------------------
# GEE image builders
# ---------------------------------------------------------------------------

def build_daily_weather_image(era5_hourly: ee.ImageCollection, day: ee.Date) -> ee.Image:
    next_day = day.advance(1, "day")
    dc = era5_hourly.filterDate(day, next_day)

    t2m = dc.select("temperature_2m").mean().rename("t2m_mean")
    u10 = dc.select("u_component_of_wind_10m").mean().rename("u10_mean")
    v10 = dc.select("v_component_of_wind_10m").mean().rename("v10_mean")
    td = dc.select("dewpoint_temperature_2m").mean().rename("td2m_mean")
    tp = dc.select("total_precipitation").sum().rename("tp_sum")

    # VPD in kPa from daily mean T and Td (both Kelvin).
    tc = t2m.subtract(273.15)
    tdc = td.subtract(273.15)
    es = tc.expression("0.6108 * exp((17.27 * T) / (T + 237.3))", {"T": tc})
    ea = tdc.expression("0.6108 * exp((17.27 * Td) / (Td + 237.3))", {"Td": tdc})
    vpd = es.subtract(ea).max(0).rename("vpd")

    return ee.Image.cat([t2m, u10, v10, tp, vpd]).set("date", day.format("YYYYMMdd"))


def build_daily_ndvi_image(ndvi_col: ee.ImageCollection, day: ee.Date) -> ee.Image:
    # MOD13Q1 is 16-day composite; use 32-day rolling window ending on day.
    ndvi = (
        ndvi_col.filterDate(day.advance(-32, "day"), day.advance(1, "day"))
        .select("NDVI")
        .median()
        .multiply(0.0001)
        .rename("ndvi")
    )
    return ndvi.set("date", day.format("YYYYMMdd"))


def build_daily_viirs_image(viirs_col: ee.ImageCollection, day: ee.Date) -> ee.Image:
    day_col = viirs_col.filterDate(day, day.advance(1, "day"))
    count = day_col.size()

    def with_data():
        img = ee.Image(day_col.mosaic())
        bands = img.bandNames()
        use_band = ee.String(
            ee.Algorithms.If(
                bands.contains("confidence"),
                "confidence",
                ee.Algorithms.If(bands.contains("T21"), "T21", bands.get(0)),
            )
        )
        return img.select([use_band]).gt(0).rename("viirs").toUint8()

    return ee.Image(
        ee.Algorithms.If(count.gt(0), with_data(), ee.Image(0).rename("viirs").toUint8())
    ).set("date", day.format("YYYYMMdd"))


# ---------------------------------------------------------------------------
# GEE raw export stage
# ---------------------------------------------------------------------------

def export_case_raw(case_cfg: dict, cfg: dict, config_dir: Path, dry_run: bool, max_days, tail_days):
    case_id = case_cfg["id"]
    out_cfg = cfg["output"]
    scale = int(out_cfg["scale_m"])
    crs = out_cfg["crs"]
    base_dir = resolve_project_path(out_cfg["base_dir"]) / case_id

    aoi = resolve_aoi(case_cfg, config_dir)
    col_cfg = cfg["collections"]

    era5_hourly = ee.ImageCollection(col_cfg["weather_hourly"]).filterBounds(aoi)
    ndvi_col = ee.ImageCollection(col_cfg["ndvi"]).filterBounds(aoi)
    viirs_col = ee.ImageCollection(col_cfg["viirs_active_fire"]).filterBounds(aoi)
    burned_col = ee.ImageCollection(col_cfg["burned_mcd64a1"]).filterBounds(aoi)

    dem = ee.Image(col_cfg["dem"]).select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    landcover = ee.ImageCollection(col_cfg["landcover"]).first().rename("landcover")

    days = select_days(case_cfg["start_date"], case_cfg["end_date"], max_days=max_days, tail_days=tail_days)

    print(f"\n[CASE] {case_id} | days={len(days)}")

    static_exports = [
        (dem, base_dir / "static" / "dem_100m.tif"),
        (slope, base_dir / "static" / "slope_100m.tif"),
        (aspect, base_dir / "static" / "aspect_100m.tif"),
        (landcover, base_dir / "static" / "landcover_100m.tif"),
    ]
    for img, path in static_exports:
        if dry_run:
            print(f"[DRY] static -> {path}")
        else:
            export_local(img, path, aoi, scale, crs)
            print(f"[SAVE] static -> {path}")

    # Monthly BurnDate rasters.
    months = sorted({d.strftime("%Y%m") for d in days})
    for ym in months:
        mstart = datetime.strptime(ym + "01", "%Y%m%d")
        mend = (mstart.replace(day=28) + timedelta(days=4)).replace(day=1)
        burn_month = (
            burned_col
            .filterDate(ee.Date(mstart.strftime("%Y-%m-%d")), ee.Date(mend.strftime("%Y-%m-%d")))
            .select("BurnDate")
            .mosaic()
            .rename("BurnDate")
        )
        out_path = base_dir / "time_series" / "burned_monthly" / f"burndate_{ym}.tif"
        if dry_run:
            print(f"[DRY] burned_monthly -> {out_path}")
        else:
            export_local(burn_month, out_path, aoi, scale, crs)
            print(f"[SAVE] burned_monthly -> {out_path}")

    # Daily time-series layers.
    for idx, d in enumerate(days, 1):
        dstr = d.strftime("%Y%m%d")
        d_ee = ee.Date(d.strftime("%Y-%m-%d"))
        
        # Progress indicator
        progress_pct = int(100.0 * idx / len(days))
        progress_bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
        print(f"\r[PROGRESS] {case_id} | Day {idx}/{len(days)} ({progress_pct}%) [{progress_bar}] {dstr}", end="", flush=True)

        weather_path = base_dir / "time_series" / "weather_daily" / f"weather_{dstr}.tif"
        ndvi_path = base_dir / "time_series" / "ndvi_daily" / f"ndvi_{dstr}.tif"
        viirs_path = base_dir / "time_series" / "viirs_daily" / f"viirs_{dstr}.tif"

        if dry_run:
            print(f"\n[DRY] weather_daily -> {weather_path}")
            print(f"[DRY] ndvi_daily -> {ndvi_path}")
            print(f"[DRY] viirs_daily -> {viirs_path}")
        else:
            export_local_with_scale_fallback(
                build_daily_weather_image(era5_hourly, d_ee),
                weather_path,
                aoi,
                scale,
                crs,
                scale_candidates=[150, 200, 250, 300],
            )
            export_local(build_daily_ndvi_image(ndvi_col, d_ee), ndvi_path, aoi, scale, crs)
            export_local(build_daily_viirs_image(viirs_col, d_ee), viirs_path, aoi, scale, crs)
    
    print()  # Newline after progress bar
    print(f"[DONE] {case_id} daily exports complete")


# ---------------------------------------------------------------------------
# Local preprocessing stage: build daily multichannel GeoTIFF stacks
# ---------------------------------------------------------------------------

def read_single_band(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
    return arr, profile, crs, transform


def reproject_to_match(src_arr, src_crs, src_transform, dst_shape, dst_crs, dst_transform, resampling=Resampling.bilinear):
    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst


def align_raster(path: Path, ref_profile: dict, is_categorical: bool, logs: list):
    arr, _, crs, transform = read_single_band(path)
    ref_shape = (ref_profile["height"], ref_profile["width"])
    ref_crs = ref_profile["crs"]
    ref_transform = ref_profile["transform"]

    if arr.shape != ref_shape or crs != ref_crs or transform != ref_transform:
        logs.append({
            "type": "alignment",
            "file": str(path),
            "message": "shape/crs/transform mismatch; reprojected to reference",
        })
        resampling = Resampling.nearest if is_categorical else Resampling.bilinear
        arr = reproject_to_match(arr, crs, transform, ref_shape, ref_crs, ref_transform, resampling=resampling)

    if np.isnan(arr).any():
        n_missing = int(np.isnan(arr).sum())
        logs.append({"type": "missing", "file": str(path), "message": f"NaN values replaced: {n_missing}"})
        arr = np.nan_to_num(arr, nan=0.0)

    return arr


def burned_mask_from_monthly(case_dir: Path, day: datetime, ref_profile: dict, logs: list):
    ym = day.strftime("%Y%m")
    burn_path = case_dir / "time_series" / "burned_monthly" / f"burndate_{ym}.tif"
    if not burn_path.exists():
        logs.append({"type": "missing", "file": str(burn_path), "message": "missing monthly BurnDate"})
        return np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32), False
    burn = align_raster(burn_path, ref_profile, is_categorical=True, logs=logs)
    day_of_year = day.timetuple().tm_yday
    mask = (burn == day_of_year).astype(np.float32)
    return mask, True


def load_band(path: Path, ref_profile: dict, logs: list, is_categorical: bool = False):
    if not path.exists():
        logs.append({"type": "missing", "file": str(path), "message": "file not found"})
        return np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32), False
    arr = align_raster(path, ref_profile, is_categorical=is_categorical, logs=logs)
    return arr.astype(np.float32), True


def load_multiband(path: Path, ref_profile: dict, logs: list):
    if not path.exists():
        logs.append({"type": "missing", "file": str(path), "message": "file not found"})
        return np.zeros((0, ref_profile["height"], ref_profile["width"]), dtype=np.float32), False

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        src_crs = src.crs
        src_transform = src.transform

    if arr.ndim != 3:
        logs.append({"type": "shape", "file": str(path), "message": f"unexpected ndim={arr.ndim}"})
        return np.zeros((0, ref_profile["height"], ref_profile["width"]), dtype=np.float32), False

    ref_shape = (ref_profile["height"], ref_profile["width"])
    ref_crs = ref_profile["crs"]
    ref_transform = ref_profile["transform"]

    if arr.shape[1:] != ref_shape or src_crs != ref_crs or src_transform != ref_transform:
        logs.append(
            {
                "type": "alignment",
                "file": str(path),
                "message": "multiband shape/crs/transform mismatch; reprojected to reference",
            }
        )
        reproj = np.zeros((arr.shape[0], ref_shape[0], ref_shape[1]), dtype=np.float32)
        for b in range(arr.shape[0]):
            reproject(
                source=arr[b],
                destination=reproj[b],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
        arr = reproj

    if np.isnan(arr).any():
        n_missing = int(np.isnan(arr).sum())
        logs.append({"type": "missing", "file": str(path), "message": f"NaN values replaced: {n_missing}"})
        arr = np.nan_to_num(arr, nan=0.0)

    return arr, True


def build_daily_stacks(case_cfg: dict, cfg: dict, max_days, tail_days, dry_run: bool):
    case_id = case_cfg["id"]
    case_dir = resolve_project_path(cfg["output"]["base_dir"]) / case_id
    out_dir = case_dir / "daily_stacks"

    all_days = select_days(case_cfg["start_date"], case_cfg["end_date"], max_days=max_days, tail_days=tail_days)

    # In dry-run mode print planned outputs without touching the filesystem.
    if dry_run:
        band_count = 5 * 3 + 5 + 3 + 1 + 1  # 25 bands total
        for d in all_days[:-1]:
            dstr = d.strftime("%Y%m%d")
            print(f"[DRY] stack -> {out_dir / f'daily_{dstr}.tif'} | bands={band_count}")
        print(f"[DRY] metadata -> {case_dir / 'metadata' / 'preprocess_log.json'}")
        return

    dem_path = case_dir / "static" / "dem_100m.tif"
    slope_path = case_dir / "static" / "slope_100m.tif"
    aspect_path = case_dir / "static" / "aspect_100m.tif"
    lc_path = case_dir / "static" / "landcover_100m.tif"

    if not dem_path.exists():
        raise FileNotFoundError(f"Missing DEM for preprocessing: {dem_path}")

    dem_arr, profile, _, _ = read_single_band(dem_path)
    profile.update(dtype=rasterio.float32, count=0, compress="lzw")

    logs = []
    dem_arr = align_raster(dem_path, profile, is_categorical=False, logs=logs)
    slope_arr = align_raster(slope_path, profile, is_categorical=False, logs=logs)
    aspect_arr = align_raster(aspect_path, profile, is_categorical=False, logs=logs)
    lc_arr = align_raster(lc_path, profile, is_categorical=True, logs=logs)

    build_days = all_days[:-1]  # skip last day (needs t+1 target)
    out_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "metadata").mkdir(parents=True, exist_ok=True)

    generated = []
    skipped = []
    WEATHER_NAMES = ["t2m_mean", "u10_mean", "v10_mean", "tp_sum", "vpd"]

    for idx, d in enumerate(build_days, 1):
        # Progress indicator
        progress_pct = int(100.0 * idx / len(build_days))
        progress_bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
        print(f"\r[PREPROCESS] {case_id} | Day {idx}/{len(build_days)} ({progress_pct}%) [{progress_bar}]", end="", flush=True)
        
        d_t = d.strftime("%Y%m%d")
        d_m1 = (d - timedelta(days=1)).strftime("%Y%m%d")
        d_m2 = (d - timedelta(days=2)).strftime("%Y%m%d")

        weather_t, ok_wt = load_multiband(case_dir / "time_series" / "weather_daily" / f"weather_{d_t}.tif", profile, logs)
        weather_m1, ok_w1 = load_multiband(case_dir / "time_series" / "weather_daily" / f"weather_{d_m1}.tif", profile, logs)
        weather_m2, ok_w2 = load_multiband(case_dir / "time_series" / "weather_daily" / f"weather_{d_m2}.tif", profile, logs)
        ndvi_t, ok_ndvi = load_band(case_dir / "time_series" / "ndvi_daily" / f"ndvi_{d_t}.tif", profile, logs)
        viirs_t, ok_viirs = load_band(case_dir / "time_series" / "viirs_daily" / f"viirs_{d_t}.tif", profile, logs, is_categorical=True)

        fire_t, ok_ft = burned_mask_from_monthly(case_dir, d, profile, logs)
        fire_m1, _ = burned_mask_from_monthly(case_dir, d - timedelta(days=1), profile, logs)
        fire_m2, _ = burned_mask_from_monthly(case_dir, d - timedelta(days=2), profile, logs)
        y_t1, ok_y = burned_mask_from_monthly(case_dir, d + timedelta(days=1), profile, logs)

        if not (ok_wt and ok_ndvi and ok_viirs and ok_ft and ok_y):
            skipped.append({
                "date": d_t,
                "reason": "critical missing inputs (weather_t, ndvi_t, viirs_t, fire_t, or y_t1)",
            })
            continue

        if weather_t.ndim != 3 or weather_t.shape[0] < 5:
            skipped.append({"date": d_t, "reason": f"weather_t has unexpected shape {weather_t.shape}"})
            continue

        def wb(arr, idx, ok):
            if ok and arr.ndim == 3 and arr.shape[0] > idx:
                return arr[idx].astype(np.float32)
            return np.zeros((profile["height"], profile["width"]), dtype=np.float32)

        ndvi_arr = (ndvi_t[0] if ndvi_t.ndim == 3 else ndvi_t).astype(np.float32)
        viirs_arr = (viirs_t[0] if viirs_t.ndim == 3 else viirs_t).astype(np.float32)

        bands = []
        band_names = []

        for i, n in enumerate(WEATHER_NAMES):
            bands.append(wb(weather_t, i, True)); band_names.append(f"{n}_t")
        for i, n in enumerate(WEATHER_NAMES):
            bands.append(wb(weather_m1, i, ok_w1)); band_names.append(f"{n}_t-1")
        for i, n in enumerate(WEATHER_NAMES):
            bands.append(wb(weather_m2, i, ok_w2)); band_names.append(f"{n}_t-2")

        bands += [ndvi_arr, dem_arr.astype(np.float32), slope_arr.astype(np.float32),
                  aspect_arr.astype(np.float32), lc_arr.astype(np.float32)]
        band_names += ["ndvi_t", "dem", "slope", "aspect", "landcover"]

        bands += [fire_t, fire_m1, fire_m2]
        band_names += ["fire_t", "fire_t-1", "fire_t-2"]

        bands.append(viirs_arr); band_names.append("viirs_t")
        bands.append(y_t1); band_names.append("y_t+1")

        stack = np.stack(bands, axis=0).astype(np.float32)

        out_path = out_dir / f"daily_{d_t}.tif"

        write_profile = profile.copy()
        write_profile.update(count=stack.shape[0], dtype=rasterio.float32)
        with rasterio.open(out_path, "w", **write_profile) as dst:
            dst.write(stack)
            dst.descriptions = tuple(band_names)
        generated.append(d_t)

    print()  # Newline after progress bar
    with (case_dir / "metadata" / "preprocess_log.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"case_id": case_id, "generated_stacks": generated, "skipped_days": skipped, "logs": logs},
            f, ensure_ascii=False, indent=2,
        )
    print(f"[DONE] {case_id} preprocessing complete: {len(generated)} stacks generated, {len(skipped)} skipped")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Final daily wildfire pipeline (collect + preprocess).")
    parser.add_argument("--config", default=str(CASES_CONFIG_PATH), help="Path to JSON config file.")
    parser.add_argument("--case", default="all", help="Case id from config, or 'all'.")
    parser.add_argument("--project", default=None, help="GEE Cloud project id (overrides config).")
    parser.add_argument("--max-days", type=int, default=None, help="Limit first N days in each case.")
    parser.add_argument("--tail-days", type=int, default=None, help="Limit to the last N days in each case.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without exporting/writing.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect raw rasters from GEE.")
    parser.add_argument("--preprocess-only", action="store_true", help="Only build daily stacks from local rasters.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    config_dir = cfg_path.parent

    gee_project = args.project or cfg.get("gee_project")
    if gee_project:
        ee.Initialize(project=gee_project)
        print(f"GEE project: {gee_project}")
    else:
        ee.Initialize()

    all_cases = cfg.get("cases", [])
    selected = all_cases if args.case == "all" else [c for c in all_cases if c.get("id") == args.case]
    if not selected:
        raise ValueError(f"No cases found for --case={args.case}")

    print(f"Selected cases: {[c['id'] for c in selected]}")
    print(f"Dry run: {args.dry_run} | Max days: {args.max_days} | Tail days: {args.tail_days}")

    if args.max_days is not None and args.tail_days is not None:
        raise ValueError("Use only one of --max-days or --tail-days.")

    do_collect = not args.preprocess_only
    do_preprocess = not args.collect_only

    for case_cfg in selected:
        case_id = case_cfg.get("id", "unknown_case")
        try:
            if do_collect:
                export_case_raw(
                    case_cfg,
                    cfg,
                    config_dir,
                    dry_run=args.dry_run,
                    max_days=args.max_days,
                    tail_days=args.tail_days,
                )
            if do_preprocess:
                build_daily_stacks(
                    case_cfg,
                    cfg,
                    max_days=args.max_days,
                    tail_days=args.tail_days,
                    dry_run=args.dry_run,
                )
        except Exception as exc:
            print(f"[ERROR] Case '{case_id}' failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
