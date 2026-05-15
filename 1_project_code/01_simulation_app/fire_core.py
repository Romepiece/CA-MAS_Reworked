from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DATA_DIR = OUTPUTS_DIR / "raw"
DATE_PATTERN = re.compile(r"(\d{8})")
EPSILON = 1e-9
STATIC_FILE_CANDIDATES = {
    "dem": ["dem_100m.tif"],
    "slope": ["slope_100m.tif"],
    "aspect": ["aspect_100m.tif"],
    "fuel": ["fuel_model_100m.tif"],
    "landcover": ["landcover_100m.tif"],
    "ndvi": ["ndvi_median_100m.tif", "ndvi_100m.tif"],
    "barrier_roads": ["barriers_roads_100m.tif"],
    "barrier_water": ["barriers_water_100m.tif"],
}
WEATHER_PREFIXES = ["u_wind", "v_wind", "wind_speed", "wind_dir", "t2m", "vpd", "rh", "tp", "td2m"]


@dataclass(slots=True)
class LegacyProjectLayout:
    project_root: Path
    data_dir: Path
    static_dir: Path
    time_series_dir: Path
    burned_dir: Path
    wind_dir: Path
    viirs_dir: Path
    ndvi_dir: Path


@dataclass(slots=True)
class RasterLayer:
    name: str
    path: Path
    array: np.ndarray
    transform: object


@dataclass(slots=True)
class ValidationFinding:
    severity: str
    code: str
    message: str


@dataclass(slots=True)
class FireCAConfig:
    base_prob: float = 0.15
    temp_sensitivity: float = 0.005
    vpd_beta: float = 0.02
    rh_beta: float = 0.01
    tp_beta: float = 2.0
    use_barriers: bool = False


@dataclass(slots=True)
class DailyFireMetrics:
    date: str
    iou: float
    dice: float
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class HybridMASConfig:
    drone_count: int = 10
    sensing_radius_px: float = 8.0
    speed_px_per_step: float = 35.0
    steps_per_day: int = 24
    target_strategy: str = "clustered_growth_frontier"
    use_spread_bias: bool = False
    assignment_method: str = "hungarian"
    assimilation_mode: str = "binary_confirm_deny"
    min_target_spacing_px: float = 16.0
    cluster_grid_px: int = 24
    max_waypoints_per_cluster: int = 4
    macro_sector_grid_px: int = 72
    diversity_alpha: float = 1.0
    diversity_beta: float = 0.6
    diversity_gamma: float = 0.8
    diversity_delta: float = 0.8
    diversity_same_sector_penalty: float = 0.9
    spread_bias_radius_px: int = 2
    spread_bias_positive_weight: float = 0.25
    spread_bias_frontier_weight: float = 0.0
    spread_bias_negative_weight: float = 0.3
    spread_bias_cap: float = 0.3
    burned_assimilation_radius_px: int = 2
    clear_assimilation_radius_px: int = 2
    weather_assimilation_radius_px: int = 6
    weather_blend_weight: float = 0.75


@dataclass(slots=True)
class WeatherPriorConfig:
    enabled: bool = True
    coarse_radius_px: int = 8
    use_global_daily_average: bool = True


@dataclass(slots=True)
class HybridDayMetrics:
    date: str
    targeting_mode: str
    cluster_count: int
    assigned_cluster_count: int
    selected_cluster_mean_pairwise_distance: float
    selected_cluster_min_pairwise_distance: float
    selected_macro_sector_count: int
    selected_connected_component_count: int
    drone_spatial_dispersion: float
    mean_cluster_size_pixels: float
    waypoints_total: int
    mean_waypoints_per_drone: float
    observed_cluster_pixels: int
    observed_cluster_coverage_ratio: float
    growth_waypoint_ratio: float
    frontier_waypoint_ratio: float
    unobserved_waypoint_ratio: float
    observed_growth_pixels: int
    growth_coverage_ratio: float
    iou_hybrid: float
    iou_ca: float
    iou_improvement: float
    frontier_iou_hybrid: float
    frontier_iou_ca: float
    frontier_iou_improvement: float
    dice: float
    precision_hybrid: float
    recall_hybrid: float
    f1_hybrid: float
    precision_ca: float
    recall_ca: float
    f1_ca: float
    frontier_f1_hybrid: float
    frontier_f1_ca: float
    frontier_f1_improvement: float
    precision_improvement: float
    recall_improvement: float
    f1_improvement: float
    growth_iou_hybrid: float
    growth_iou_ca: float
    growth_iou_improvement: float
    growth_f1_hybrid: float
    growth_f1_ca: float
    growth_f1_improvement: float
    true_growth_pixels: int
    predicted_growth_hybrid_pixels: int
    predicted_growth_ca_pixels: int
    raw_predicted_growth_pixels: int
    filtered_growth_target_pixels: int
    frontier_candidate_pixels: int
    selected_target_pixels: int
    assigned_target_pixels: int
    observed_footprint_pixels: int
    observed_burned_pixels: int
    observed_clear_pixels: int
    observed_frontier_pixels: int
    frontier_pixels: int
    frontier_coverage_ratio: float
    corrected_to_burned_pixels: int
    corrected_to_clear_pixels: int
    corrected_total_pixels: int


@dataclass(slots=True)
class ExportPaths:
    case_name: str
    raw_case_dir: Path
    static_dir: Path
    time_series_dir: Path
    burned_dir: Path
    viirs_dir: Path
    weather_dir: Path
    metadata_dir: Path


@dataclass(slots=True)
class GEEExportConfig:
    project_id: str
    raw_data_dir: str
    crs: str
    grid_resolution_m: int
    start_date: str
    end_date: str
    aoi_backend: str = "geojson"
    earth_engine_asset_id: str | None = None
    export_static_layers: bool = True
    export_time_series: bool = True


@dataclass(slots=True)
class TargetCluster:
    cluster_id: str
    cells: list[tuple[int, int]]
    centroid: tuple[int, int]
    waypoints: list[tuple[int, int]]
    score: float
    source: str
    grid_key: tuple[int, int]
    macro_sector_id: tuple[int, int]
    component_id: int


def _smooth_field(array: np.ndarray, radius: int) -> np.ndarray:
    field = np.asarray(array, dtype=float)
    if radius <= 0:
        return field.copy()
    smoothed = np.zeros_like(field, dtype=float)
    counts = np.zeros_like(field, dtype=float)
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            shifted = np.roll(np.roll(field, row_offset, axis=0), col_offset, axis=1)
            valid = np.ones_like(field, dtype=float)
            if row_offset > 0:
                valid[:row_offset, :] = 0.0
            elif row_offset < 0:
                valid[row_offset:, :] = 0.0
            if col_offset > 0:
                valid[:, :col_offset] = 0.0
            elif col_offset < 0:
                valid[:, col_offset:] = 0.0
            smoothed += shifted * valid
            counts += valid
    return smoothed / np.maximum(counts, 1.0)


def _broadcast_scalar_field(value: float, reference_shape: tuple[int, ...]) -> np.ndarray:
    return np.full(reference_shape, float(value), dtype=float)


def _build_global_daily_prior(daily: dict[str, np.ndarray], reference_shape: np.ndarray) -> dict[str, np.ndarray]:
    prior: dict[str, np.ndarray] = {}

    wind_speed = np.asarray(daily.get("wind_speed", np.zeros_like(reference_shape, dtype=float)), dtype=float)
    wind_dir = np.asarray(daily.get("wind_dir", np.zeros_like(reference_shape, dtype=float)), dtype=float)
    u_component = wind_speed * np.cos(wind_dir)
    v_component = wind_speed * np.sin(wind_dir)
    mean_u = float(np.mean(u_component))
    mean_v = float(np.mean(v_component))
    mean_wind_speed = float(np.hypot(mean_u, mean_v))
    mean_wind_dir = float(np.arctan2(mean_v, mean_u)) if mean_wind_speed > EPSILON else 0.0

    prior["wind_speed"] = _broadcast_scalar_field(mean_wind_speed, reference_shape.shape)
    prior["wind_dir"] = _broadcast_scalar_field(mean_wind_dir, reference_shape.shape)

    for key in ("t2m", "vpd", "rh", "tp", "td2m"):
        field = np.asarray(daily.get(key, np.zeros_like(reference_shape, dtype=float)), dtype=float)
        prior[key] = _broadcast_scalar_field(float(np.mean(field)), reference_shape.shape)

    return prior


def _derive_weather_fields(daily: dict[str, np.ndarray], reference_shape: np.ndarray) -> dict[str, np.ndarray]:
    zeros = np.zeros_like(reference_shape, dtype=float)
    derived = {key: np.asarray(value, dtype=float) for key, value in daily.items()}

    if "wind_speed" not in derived:
        u_component = np.asarray(derived.get("u_wind", derived.get("u10", derived.get("u10_mean", zeros))), dtype=float)
        v_component = np.asarray(derived.get("v_wind", derived.get("v10", derived.get("v10_mean", zeros))), dtype=float)
        derived["wind_speed"] = np.hypot(u_component, v_component)
        derived["wind_dir"] = np.arctan2(v_component, u_component)
    elif "wind_dir" not in derived:
        derived["wind_dir"] = np.zeros_like(derived["wind_speed"], dtype=float)

    if "t2m" not in derived:
        derived["t2m"] = np.asarray(derived.get("t2m_mean", zeros), dtype=float)
    if "tp" not in derived:
        derived["tp"] = np.asarray(derived.get("tp_sum", zeros), dtype=float)
    if "td2m" not in derived:
        derived["td2m"] = np.asarray(derived.get("td2m_mean", derived.get("t2m", zeros)), dtype=float)
    if "vpd" not in derived:
        derived["vpd"] = np.zeros_like(reference_shape, dtype=float)

    if "rh" not in derived:
        temp_c = np.asarray(derived["t2m"], dtype=float) - 273.15
        dewpoint_c = np.asarray(derived["td2m"], dtype=float) - 273.15
        es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3 + EPSILON))
        ea = 0.6108 * np.exp((17.27 * dewpoint_c) / (dewpoint_c + 237.3 + EPSILON))
        derived["rh"] = np.clip(100.0 * ea / np.maximum(es, EPSILON), 0.0, 100.0)

    return derived


def _build_weather_prior(
    weather_data: dict[str, dict[str, np.ndarray]],
    reference_shape: np.ndarray,
    prior_config: WeatherPriorConfig | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    config = prior_config or WeatherPriorConfig()
    sensor_weather: dict[str, dict[str, np.ndarray]] = {}
    prior_weather: dict[str, dict[str, np.ndarray]] = {}
    for date_str, daily in weather_data.items():
        derived = _derive_weather_fields(daily, reference_shape)
        sensor_weather[date_str] = {key: np.asarray(value, dtype=float).copy() for key, value in derived.items()}
        if config.enabled:
            if config.use_global_daily_average:
                prior_weather[date_str] = _build_global_daily_prior(derived, reference_shape)
            else:
                prior_weather[date_str] = {key: _smooth_field(value, config.coarse_radius_px) for key, value in derived.items()}
        else:
            prior_weather[date_str] = {key: np.asarray(value, dtype=float).copy() for key, value in derived.items()}
    return prior_weather, sensor_weather


def _assimilate_local_weather(
    prior_weather: dict[str, np.ndarray],
    sensor_weather: dict[str, np.ndarray],
    observed_cells: set[tuple[int, int]],
    blend_weight: float,
    radius: int,
) -> dict[str, np.ndarray]:
    assimilated = {key: np.asarray(value, dtype=float).copy() for key, value in prior_weather.items()}
    if not observed_cells:
        return assimilated

    footprint_mask = np.zeros_like(next(iter(assimilated.values())), dtype=bool)
    for y, x in observed_cells:
        footprint_mask[y, x] = True
    influence_mask = _dilate_binary_mask(footprint_mask, radius=max(radius, 0))

    weight = float(np.clip(blend_weight, 0.0, 1.0))
    for key, field in assimilated.items():
        sensor_field = np.asarray(sensor_weather.get(key, field), dtype=float)
        field[influence_mask] = (1.0 - weight) * field[influence_mask] + weight * sensor_field[influence_mask]
    return assimilated


@dataclass(slots=True)
class CaseDefinition:
    name: str
    aoi_source: str
    start_date: str
    end_date: str
    notes: str = ""


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_legacy_project_layout(project_root: str | Path) -> LegacyProjectLayout:
    root = Path(project_root).resolve()
    data_dir = root / "data"
    return LegacyProjectLayout(
        project_root=root,
        data_dir=data_dir,
        static_dir=data_dir / "static",
        time_series_dir=data_dir / "time_series",
        burned_dir=data_dir / "time_series" / "burned",
        wind_dir=data_dir / "time_series" / "wind",
        viirs_dir=data_dir / "time_series" / "viirs",
        ndvi_dir=data_dir / "time_series" / "ndvi",
    )


def extract_date(filename: str) -> str | None:
    match = DATE_PATTERN.search(filename)
    return match.group(1) if match else None


def read_single_band_raster(path: str | Path) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as src:
        return src.read(1), src.transform


def match_shape(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_rows, ref_cols = reference.shape
    tgt_rows, tgt_cols = target.shape
    return target[: min(ref_rows, tgt_rows), : min(ref_cols, tgt_cols)]


def _find_existing_file(directory: Path, candidates: list[str]) -> Path | None:
    for filename in candidates:
        path = directory / filename
        if path.exists():
            return path
    return None


def load_static_layers(layout: LegacyProjectLayout) -> dict[str, RasterLayer]:
    layers: dict[str, RasterLayer] = {}
    dem_path = _find_existing_file(layout.static_dir, STATIC_FILE_CANDIDATES["dem"])
    if dem_path is None:
        raise FileNotFoundError(f"Required DEM layer not found in {layout.static_dir}")
    dem_array, dem_transform = read_single_band_raster(dem_path)
    layers["dem"] = RasterLayer("dem", dem_path, dem_array, dem_transform)

    for layer_name, candidates in STATIC_FILE_CANDIDATES.items():
        if layer_name == "dem":
            continue
        layer_path = _find_existing_file(layout.static_dir, candidates)
        if layer_path is None:
            continue
        array, transform = read_single_band_raster(layer_path)
        array = match_shape(dem_array, array)
        layers[layer_name] = RasterLayer(layer_name, layer_path, array, transform)

    if "fuel" not in layers and "landcover" in layers:
        landcover = layers["landcover"].array
        fuel_model = np.zeros_like(landcover, dtype=float)
        mapping = {10: 0.8, 20: 0.7, 30: 0.6, 40: 0.2, 50: 0.1, 60: 0.0, 70: 0.0, 80: 0.0, 90: 0.0, 95: 0.1, 100: 0.0}
        for class_id, fuel_value in mapping.items():
            fuel_model[landcover == class_id] = fuel_value
        layers["fuel"] = RasterLayer("fuel", layers["landcover"].path, fuel_model, layers["landcover"].transform)

    if "barrier_roads" in layers or "barrier_water" in layers:
        barrier = np.zeros_like(dem_array, dtype=np.uint8)
        if "barrier_roads" in layers:
            barrier = np.clip(barrier + match_shape(dem_array, layers["barrier_roads"].array), 0, 1)
        if "barrier_water" in layers:
            barrier = np.clip(barrier + match_shape(dem_array, layers["barrier_water"].array), 0, 1)
        layers["barrier"] = RasterLayer("barrier", layout.static_dir / "derived_barrier", barrier, layers["dem"].transform)

    return layers


def load_burned_series(layout: LegacyProjectLayout) -> tuple[list[np.ndarray], list[str], list[Path]]:
    burned_files = sorted(layout.burned_dir.glob("burned_*.tif"), key=lambda path: extract_date(path.name) or "")
    if not burned_files:
        raise FileNotFoundError(f"No burned rasters found in {layout.burned_dir}")
    series: list[np.ndarray] = []
    dates: list[str] = []
    for path in burned_files:
        array, _ = read_single_band_raster(path)
        series.append((array > 0).astype(np.uint8))
        dates.append(extract_date(path.name) or "")
    return series, dates, burned_files


def load_viirs_series(layout: LegacyProjectLayout, reference_shape: np.ndarray, burned_dates: list[str]) -> list[np.ndarray]:
    viirs_files = sorted(layout.viirs_dir.glob("viirs_*.tif"), key=lambda path: extract_date(path.name) or "")
    viirs_by_date: dict[str, np.ndarray] = {}
    for path in viirs_files:
        date_str = extract_date(path.name)
        if not date_str:
            continue
        array, _ = read_single_band_raster(path)
        viirs_by_date[date_str] = match_shape(reference_shape, (array > 0).astype(np.uint8))
    return [viirs_by_date.get(date_str, np.zeros_like(reference_shape, dtype=np.uint8)) for date_str in burned_dates]


def build_initial_fire_from_viirs(viirs_fire: np.ndarray, dilation_radius: int = 3) -> np.ndarray:
    viirs_mask = (np.asarray(viirs_fire) > 0).astype(np.uint8)
    if int(viirs_mask.sum()) == 0:
        return viirs_mask
    expanded = _dilate_binary_mask(viirs_mask.astype(bool), radius=max(dilation_radius, 0))
    return expanded.astype(np.uint8)


def build_initial_fire_from_viirs_history(
    viirs_series: list[np.ndarray],
    end_idx: int,
    history_days: int | None = None,
    dilation_radius: int = 3,
) -> np.ndarray:
    if not viirs_series:
        raise ValueError("viirs_series must not be empty")
    if end_idx < 0 or end_idx >= len(viirs_series):
        raise IndexError("end_idx is out of bounds for viirs_series")

    start_idx = 0 if history_days is None else max(0, end_idx - history_days + 1)
    nonempty_history = [
        (np.asarray(viirs_fire) > 0).astype(np.uint8)
        for viirs_fire in viirs_series[start_idx : end_idx + 1]
        if int(np.asarray(viirs_fire).sum()) > 0
    ]
    if not nonempty_history:
        return np.zeros_like(viirs_series[end_idx], dtype=np.uint8)

    merged = np.logical_or.reduce([mask > 0 for mask in nonempty_history])
    return build_initial_fire_from_viirs(merged.astype(np.uint8), dilation_radius=dilation_radius)


def discover_weather_files(layout: LegacyProjectLayout) -> dict[str, dict[str, Path]]:
    discovered: dict[str, dict[str, Path]] = {prefix: {} for prefix in WEATHER_PREFIXES}
    if not layout.wind_dir.exists():
        return discovered
    for path in layout.wind_dir.glob("*.tif"):
        date_str = extract_date(path.name)
        if not date_str:
            continue
        for prefix in WEATHER_PREFIXES:
            if path.name.startswith(f"{prefix}_"):
                discovered[prefix][date_str] = path
                break
    return discovered


def build_weather_data_by_date(
    layout: LegacyProjectLayout,
    reference_shape: np.ndarray,
    burned_dates: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    weather_files = discover_weather_files(layout)
    weather_data: dict[str, dict[str, np.ndarray]] = {}
    for date_str in burned_dates:
        daily: dict[str, np.ndarray] = {}
        for prefix in WEATHER_PREFIXES:
            path = weather_files.get(prefix, {}).get(date_str)
            if path is None:
                daily[prefix] = np.zeros_like(reference_shape, dtype=float)
                continue
            array, _ = read_single_band_raster(path)
            daily[prefix] = match_shape(reference_shape, array)
        weather_data[date_str] = daily
    return weather_data


def to_binary_mask(array: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return (np.asarray(array) > threshold).astype(np.uint8)


def clip_to_unit_interval(array: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(array, dtype=float), 0.0, 1.0)


def build_combined_barrier_mask(static_layers: dict[str, RasterLayer]) -> np.ndarray | None:
    barrier_parts = []
    for key in ("barrier_roads", "barrier_water"):
        if key in static_layers:
            barrier_parts.append(to_binary_mask(static_layers[key].array))
    if not barrier_parts:
        return None
    stacked = np.sum(barrier_parts, axis=0)
    return np.clip(stacked, 0, 1).astype(np.uint8)


def normalize_static_layers(static_layers: dict[str, RasterLayer]) -> dict[str, RasterLayer]:
    normalized = dict(static_layers)
    for key in ("barrier_roads", "barrier_water"):
        if key in normalized:
            layer = normalized[key]
            normalized[key] = RasterLayer(layer.name, layer.path, to_binary_mask(layer.array), layer.transform)
    if "ndvi" in normalized:
        layer = normalized["ndvi"]
        normalized["ndvi"] = RasterLayer(layer.name, layer.path, clip_to_unit_interval(layer.array), layer.transform)
    combined_barrier = build_combined_barrier_mask(normalized)
    if combined_barrier is not None and "dem" in normalized:
        normalized["barrier"] = RasterLayer(
            "barrier",
            normalized["dem"].path.parent / "derived_barrier",
            combined_barrier,
            normalized["dem"].transform,
        )
    return normalized


def _coverage(array: np.ndarray) -> float:
    return float(np.asarray(array).mean())


def validate_static_layers(static_layers: dict[str, RasterLayer]) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for key in ("barrier_roads", "barrier_water", "barrier"):
        if key not in static_layers:
            continue
        coverage = _coverage(static_layers[key].array)
        if coverage >= 0.999:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    code=f"{key}_full_coverage",
                    message=f"{key} covers nearly the entire grid (mean={coverage:.4f}); this usually indicates an invalid rasterization or inverted mask.",
                )
            )
        elif coverage <= 0.0001:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    code=f"{key}_empty",
                    message=f"{key} is almost empty (mean={coverage:.4f}); verify whether the barrier source was rasterized correctly.",
                )
            )

    if "ndvi" in static_layers:
        ndvi = static_layers["ndvi"].array
        if float(np.nanmin(ndvi)) < 0.0 or float(np.nanmax(ndvi)) > 1.0:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    code="ndvi_out_of_unit_interval",
                    message="NDVI lies outside [0, 1]; check whether the layer is raw scaled values and needs normalization.",
                )
            )

    return findings


def summarize_legacy_project_data(layout: LegacyProjectLayout) -> dict[str, object]:
    static_layers = load_static_layers(layout)
    burned_series, burned_dates, burned_files = load_burned_series(layout)
    weather_files = discover_weather_files(layout)
    weather_data = build_weather_data_by_date(layout, static_layers["dem"].array, burned_dates)
    return {
        "project_root": str(layout.project_root),
        "static_layers": sorted(static_layers.keys()),
        "static_shape": list(static_layers["dem"].array.shape),
        "burned_file_count": len(burned_files),
        "burned_date_start": burned_dates[0],
        "burned_date_end": burned_dates[-1],
        "weather_file_counts": {name: len(files) for name, files in weather_files.items() if files},
        "weather_days_loaded": len(weather_data),
        "barrier_coverage_mean": float(static_layers["barrier"].array.mean()) if "barrier" in static_layers else 0.0,
        "positive_pixels_first_burned_day": int(burned_series[0].sum()),
    }


def load_project_data(project_root: str | Path) -> dict[str, object]:
    layout = resolve_legacy_project_layout(project_root)
    raw_static_layers = load_static_layers(layout)
    static_layers = normalize_static_layers(raw_static_layers)
    burned_series, burned_dates, _ = load_burned_series(layout)
    viirs_series = load_viirs_series(layout, static_layers["dem"].array, burned_dates)
    weather_data = build_weather_data_by_date(layout, static_layers["dem"].array, burned_dates)
    return {
        "layout": layout,
        "static_layers": static_layers,
        "static_arrays": {name: layer.array for name, layer in static_layers.items()},
        "burned_series": burned_series,
        "burned_dates": burned_dates,
        "viirs_series": viirs_series,
        "weather_data": weather_data,
        "validation_findings": validate_static_layers(static_layers),
    }


class FireCA:
    def __init__(
        self,
        static_data: dict[str, np.ndarray],
        weather_data: dict[str, dict[str, np.ndarray]],
        config: FireCAConfig | None = None,
    ) -> None:
        self.config = config or FireCAConfig()
        self.dem = np.asarray(static_data["dem"], dtype=float)
        self.slope = np.asarray(static_data["slope"], dtype=float)
        self.fuel = np.asarray(static_data["fuel"], dtype=float)
        self.barrier = np.asarray(static_data.get("barrier", np.zeros_like(self.dem)), dtype=np.uint8)
        self.weather = weather_data
        self.spread_bias_by_date: dict[str, np.ndarray] = {}

        shape = self.dem.shape
        for name, array in static_data.items():
            if np.asarray(array).shape != shape:
                raise ValueError(f"Layer {name} has mismatched shape {np.asarray(array).shape} != {shape}")

        self._fuel_denominator = float(self.fuel.max()) + 1e-6

    def spread_step(self, current_fire: np.ndarray, date_str: str, rng: np.random.Generator) -> np.ndarray:
        weather = self.weather.get(date_str, {})
        spread_bias = np.asarray(self.spread_bias_by_date.get(date_str, np.zeros_like(self.dem, dtype=float)), dtype=float)
        zeros = np.zeros_like(self.dem, dtype=float)
        wind_speed = np.asarray(weather.get("wind_speed", zeros), dtype=float)
        wind_dir = np.asarray(weather.get("wind_dir", zeros), dtype=float)
        t2m = np.asarray(weather.get("t2m", zeros), dtype=float)
        vpd = np.asarray(weather.get("vpd", zeros), dtype=float)
        rh = np.asarray(weather.get("rh", zeros), dtype=float)
        tp = np.asarray(weather.get("tp", zeros), dtype=float)

        temp_c = t2m - 273.15
        current_fire = (np.asarray(current_fire) > 0).astype(np.uint8)
        new_fire = current_fire.copy()
        rows, cols = current_fire.shape

        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if current_fire[row, col] == 1:
                    continue
                neighborhood = current_fire[row - 1 : row + 2, col - 1 : col + 2]
                if int(neighborhood.sum()) == 0:
                    continue
                slope_factor = 1 + (self.slope[row, col] / 100)
                wind_factor = 1 + (wind_speed[row, col] / 10)
                fuel_factor = self.fuel[row, col] / self._fuel_denominator
                temp_factor = 1 + self.config.temp_sensitivity * max(temp_c[row, col] - 20, 0)
                vpd_factor = 1 + self.config.vpd_beta * max(vpd[row, col], 0)
                rh_factor = 1 / (1 + self.config.rh_beta * max(rh[row, col], 0))
                tp_factor = float(np.exp(-self.config.tp_beta * max(tp[row, col], 0)))
                barrier_factor = self._barrier_factor(row, col)
                wind_direction_factor = self._wind_direction_factor(neighborhood, wind_speed[row, col], wind_dir[row, col])

                probability = self.config.base_prob * slope_factor * wind_factor * fuel_factor * temp_factor
                probability *= vpd_factor * rh_factor * tp_factor * barrier_factor * wind_direction_factor
                probability *= max(0.0, 1.0 + float(spread_bias[row, col]))
                probability = min(1.0, max(0.0, probability))

                if rng.random() < probability:
                    new_fire[row, col] = 1

        return new_fire

    def set_spread_bias(self, date_str: str, bias_field: np.ndarray) -> None:
        self.spread_bias_by_date[date_str] = np.asarray(bias_field, dtype=float).copy()

    def rollout(self, initial_fire: np.ndarray, dates: list[str], seed: int | None = None) -> list[np.ndarray]:
        if not dates:
            return []
        rng = np.random.default_rng(seed)
        fire_state = (np.asarray(initial_fire) > 0).astype(np.uint8)
        predictions = [fire_state.copy()]
        for date_str in dates:
            fire_state = self.spread_step(fire_state, date_str, rng=rng)
            predictions.append(fire_state.copy())
        return predictions

    def _barrier_factor(self, row: int, col: int) -> float:
        if not self.config.use_barriers:
            return 1.0
        return 0.0 if self.barrier[row, col] > 0 else 1.0

    @staticmethod
    def _wind_direction_factor(neighborhood: np.ndarray, wind_speed: float, wind_dir: float) -> float:
        if wind_speed <= 0.1:
            return 1.0
        fire_y, fire_x = np.where(neighborhood > 0)
        if len(fire_y) == 0:
            return 1.0
        center_y = float(np.mean(fire_y) - 1)
        center_x = float(np.mean(fire_x) - 1)
        spread_dir = float(np.arctan2(-center_y, -center_x))
        wind_alignment = float(np.cos(spread_dir - wind_dir))
        return 1.0 + 0.3 * max(0.0, wind_alignment) * min(wind_speed / 5.0, 1.0)


def compute_fire_metrics(
    predicted_fire: np.ndarray,
    true_fire: np.ndarray,
    evaluation_mask: np.ndarray | None = None,
) -> DailyFireMetrics:
    predicted_fire = (np.asarray(predicted_fire) > 0).astype(np.uint8)
    true_fire = (np.asarray(true_fire) > 0).astype(np.uint8)
    if evaluation_mask is not None:
        mask = np.asarray(evaluation_mask, dtype=bool)
        predicted_fire = np.logical_and(predicted_fire > 0, mask).astype(np.uint8)
        true_fire = np.logical_and(true_fire > 0, mask).astype(np.uint8)
    intersection = int(np.logical_and(predicted_fire, true_fire).sum())
    union = int(np.logical_or(predicted_fire, true_fire).sum())
    predicted_sum = int(predicted_fire.sum())
    true_sum = int(true_fire.sum())
    false_positive = int(np.logical_and(predicted_fire == 1, true_fire == 0).sum())
    false_negative = int(np.logical_and(predicted_fire == 0, true_fire == 1).sum())

    iou = intersection / (union + EPSILON)
    dice = 2 * intersection / (predicted_sum + true_sum + EPSILON)
    precision = intersection / (intersection + false_positive + EPSILON)
    recall = intersection / (intersection + false_negative + EPSILON)
    f1 = 2 * (precision * recall) / (precision + recall + EPSILON)
    return DailyFireMetrics(date="", iou=iou, dice=dice, precision=precision, recall=recall, f1=f1)


def build_frontier_evaluation_mask(
    previous_true_fire: np.ndarray,
    current_true_fire: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    previous_true = np.asarray(previous_true_fire) > 0
    current_true = np.asarray(current_true_fire) > 0
    true_growth = np.logical_and(current_true, np.logical_not(previous_true))
    if not np.any(true_growth):
        return true_growth.astype(bool)
    return _dilate_binary_mask(true_growth, radius=max(radius, 0))


def build_daily_growth_mask(previous_fire: np.ndarray, current_fire: np.ndarray) -> np.ndarray:
    previous_state = np.asarray(previous_fire) > 0
    current_state = np.asarray(current_fire) > 0
    return np.logical_and(current_state, np.logical_not(previous_state))


def _count_binary_cells(mask: np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def _build_next_day_spread_bias(
    reference_shape: tuple[int, int],
    observed_growth_mask: np.ndarray,
    observed_frontier_burned_mask: np.ndarray,
    observed_clear_growth_mask: np.ndarray,
    config: HybridMASConfig,
) -> np.ndarray:
    bias = np.zeros(reference_shape, dtype=float)
    radius = max(int(config.spread_bias_radius_px), 0)

    if np.any(observed_growth_mask):
        bias += config.spread_bias_positive_weight * _dilate_binary_mask(observed_growth_mask, radius=radius).astype(float)
    if np.any(observed_frontier_burned_mask):
        bias += config.spread_bias_frontier_weight * _dilate_binary_mask(observed_frontier_burned_mask, radius=radius).astype(float)
    if np.any(observed_clear_growth_mask):
        bias -= config.spread_bias_negative_weight * _dilate_binary_mask(observed_clear_growth_mask, radius=radius).astype(float)

    return np.clip(bias, -config.spread_bias_cap, config.spread_bias_cap)


def evaluate_rollout(
    predicted_series: list[np.ndarray],
    true_series: list[np.ndarray],
    evaluation_dates: list[str],
) -> list[DailyFireMetrics]:
    if len(predicted_series) != len(true_series):
        raise ValueError("Predicted and true series must have equal length")
    if len(evaluation_dates) != len(true_series):
        raise ValueError("Evaluation dates and true series must have equal length")

    metrics: list[DailyFireMetrics] = []
    for date_str, predicted_fire, true_fire in zip(evaluation_dates, predicted_series, true_series, strict=True):
        daily_metrics = compute_fire_metrics(predicted_fire, true_fire)
        metrics.append(DailyFireMetrics(date=date_str, iou=daily_metrics.iou, dice=daily_metrics.dice, precision=daily_metrics.precision, recall=daily_metrics.recall, f1=daily_metrics.f1))
    return metrics


def summarize_metrics(metrics: list[DailyFireMetrics]) -> dict[str, float]:
    if not metrics:
        return {"mean_iou": 0.0, "mean_dice": 0.0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0}
    return {
        "mean_iou": float(np.mean([metric.iou for metric in metrics])),
        "mean_dice": float(np.mean([metric.dice for metric in metrics])),
        "mean_precision": float(np.mean([metric.precision for metric in metrics])),
        "mean_recall": float(np.mean([metric.recall for metric in metrics])),
        "mean_f1": float(np.mean([metric.f1 for metric in metrics])),
    }


class FireFrontPredictor:
    def __init__(self, fire_ca: FireCA) -> None:
        self.fire_ca = fire_ca

    def predict_next_step(self, current_fire: np.ndarray, date_str: str, rng: np.random.Generator) -> np.ndarray:
        return self.fire_ca.spread_step(current_fire, date_str, rng=rng)

    def identify_fire_front(self, fire_map: np.ndarray, buffer: int = 3) -> set[tuple[int, int]]:
        fire_mask = np.asarray(fire_map, dtype=bool)
        dilated = fire_mask.copy()
        for _ in range(buffer):
            padded = np.pad(dilated, 1, mode="constant", constant_values=False)
            neighbors = []
            for row_offset in range(3):
                for col_offset in range(3):
                    neighbors.append(padded[row_offset : row_offset + dilated.shape[0], col_offset : col_offset + dilated.shape[1]])
            dilated = np.logical_or.reduce(neighbors)
        front = np.logical_and(dilated, np.logical_not(fire_mask))
        return set(zip(*np.where(front), strict=False))


class TrajectoryOptimizer:
    @staticmethod
    def compute_cost_matrix(drone_positions: list[tuple[int, int]], targets: list[tuple[int, int]]) -> np.ndarray:
        cost_matrix = np.zeros((len(drone_positions), len(targets)), dtype=float)
        for drone_index, (drone_y, drone_x) in enumerate(drone_positions):
            for target_index, (target_y, target_x) in enumerate(targets):
                cost_matrix[drone_index, target_index] = np.hypot(drone_y - target_y, drone_x - target_x)
        return cost_matrix

    @staticmethod
    def optimize_assignment(drone_positions: list[tuple[int, int]], targets: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not drone_positions or not targets:
            return []
        cost_matrix = TrajectoryOptimizer.compute_cost_matrix(drone_positions, targets)
        if linear_sum_assignment is not None:
            drone_indices, target_indices = linear_sum_assignment(cost_matrix)
            return list(zip(drone_indices.tolist(), target_indices.tolist(), strict=True))

        pair_count = min(cost_matrix.shape[0], cost_matrix.shape[1])
        available_targets = set(range(cost_matrix.shape[1]))
        greedy_pairs: list[tuple[int, int]] = []
        for drone_index in range(pair_count):
            target_order = np.argsort(cost_matrix[drone_index])
            for target_index in target_order.tolist():
                if target_index in available_targets:
                    available_targets.remove(target_index)
                    greedy_pairs.append((drone_index, target_index))
                    break
        return greedy_pairs


class DroneAdvanced:
    def __init__(self, drone_id: str, y: int, x: int, vision: float = 8.0, move: float = 35.0, color: str = "cyan") -> None:
        self.id = drone_id
        self.y = float(y)
        self.x = float(x)
        self.vision = vision
        self.move = move
        self.color = color
        self.path: list[tuple[float, float]] = [(float(y), float(x))]
        self.daily_path: list[tuple[float, float]] = [(float(y), float(x))]
        self.last_step_path: list[tuple[float, float]] = [(float(y), float(x))]
        self.observed: set[tuple[int, int]] = set()
        self.assigned_target: Optional[tuple[int, int]] = None
        self.assigned_waypoints: list[tuple[int, int]] = []
        self.assigned_cluster_cells: set[tuple[int, int]] = set()
        self.adaptive_vision = vision

    def reset_daily_path(self) -> None:
        self.daily_path = [(self.y, self.x)]
        self.last_step_path = [(self.y, self.x)]

    def clear_assignment(self) -> None:
        self.assigned_target = None
        self.assigned_waypoints = []
        self.assigned_cluster_cells = set()

    def set_target(self, target: Optional[tuple[int, int]]) -> None:
        self.assigned_target = target
        self.assigned_waypoints = [target] if target is not None else []
        self.assigned_cluster_cells = set()

    def set_cluster_patrol(self, cluster_cells: list[tuple[int, int]], waypoints: list[tuple[int, int]]) -> None:
        self.assigned_cluster_cells = set(cluster_cells)
        self.assigned_waypoints = [(int(y), int(x)) for y, x in waypoints]
        self.assigned_target = self.assigned_waypoints[0] if self.assigned_waypoints else None

    def _update_adaptive_vision(self, fire_density: float) -> None:
        min_vision = max(3.0, self.vision * 0.7)
        max_vision = max(min_vision, self.vision * 1.3)
        if fire_density > 0.1:
            self.adaptive_vision = min_vision
        else:
            self.adaptive_vision = max_vision

    def _move_towards_point(self, target_y: float, target_x: float, remaining_move: float) -> float:
        dy = target_y - self.y
        dx = target_x - self.x
        distance = float(np.hypot(dy, dx))
        if distance <= EPSILON:
            self.y = float(target_y)
            self.x = float(target_x)
            return remaining_move
        travel = min(remaining_move, distance)
        self.y = float(self.y + (dy / distance) * travel)
        self.x = float(self.x + (dx / distance) * travel)
        if travel >= distance - EPSILON:
            self.y = float(target_y)
            self.x = float(target_x)
        return remaining_move - travel

    def move_optimized(
        self,
        fire_map: np.ndarray,
        shared_knowledge: set[tuple[int, int]],
        wind_speed: np.ndarray,
        wind_dir: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        step_path: list[tuple[float, float]] = [(self.y, self.x)]
        if self.assigned_waypoints:
            remaining_move = float(self.move)
            while self.assigned_waypoints and remaining_move > EPSILON:
                waypoint_y, waypoint_x = self.assigned_waypoints[0]
                remaining_move = self._move_towards_point(float(waypoint_y), float(waypoint_x), remaining_move)
                if np.hypot(self.y - waypoint_y, self.x - waypoint_x) <= 1.0:
                    self.y = float(waypoint_y)
                    self.x = float(waypoint_x)
                    self.assigned_waypoints.pop(0)
                    step_path.append((self.y, self.x))
                else:
                    break
            self.assigned_target = self.assigned_waypoints[0] if self.assigned_waypoints else None
        elif self.assigned_cluster_cells:
            cluster_unobserved = [cell for cell in self.assigned_cluster_cells if cell not in shared_knowledge]
            if cluster_unobserved:
                targets = np.asarray(cluster_unobserved, dtype=float)
                effective_distance = np.hypot(targets[:, 0] - self.y, targets[:, 1] - self.x)
                target_index = int(np.argmin(effective_distance))
                self._move_towards_point(float(targets[target_index][0]), float(targets[target_index][1]), float(self.move))
                step_path.append((self.y, self.x))
            else:
                self.clear_assignment()
        else:
            unexplored = [(int(y), int(x)) for y, x in np.argwhere(fire_map > 0) if (int(y), int(x)) not in shared_knowledge]
            if unexplored:
                targets = np.asarray(unexplored, dtype=float)
                effective_distance = np.hypot(targets[:, 0] - self.y, targets[:, 1] - self.x)
                cy, cx = int(self.y), int(self.x)
                if 0 <= cy < wind_speed.shape[0] and 0 <= cx < wind_speed.shape[1]:
                    local_wind_speed = float(wind_speed[cy, cx])
                    local_wind_dir = float(wind_dir[cy, cx])
                    if local_wind_speed > 0.1:
                        wind_vec_y = np.sin(local_wind_dir)
                        wind_vec_x = np.cos(local_wind_dir)
                        vec_y = targets[:, 0] - self.y
                        vec_x = targets[:, 1] - self.x
                        vec_norm = np.hypot(vec_y, vec_x) + EPSILON
                        alignment = (vec_y * wind_vec_y + vec_x * wind_vec_x) / vec_norm
                        effective_distance *= 1 - 0.3 * np.maximum(0.0, alignment)

                target_index = int(np.argmin(effective_distance))
                target_y, target_x = targets[target_index]
                self.y = float(np.clip(self.y + np.sign(target_y - self.y) * self.move, 0, fire_map.shape[0] - 1))
                self.x = float(np.clip(self.x + np.sign(target_x - self.x) * self.move, 0, fire_map.shape[1] - 1))
                step_path.append((self.y, self.x))
            else:
                self.y = float(np.clip(self.y + rng.integers(-int(self.move), int(self.move) + 1), 0, fire_map.shape[0] - 1))
                self.x = float(np.clip(self.x + rng.integers(-int(self.move), int(self.move) + 1), 0, fire_map.shape[1] - 1))
                step_path.append((self.y, self.x))

        if step_path[-1] != (self.y, self.x):
            step_path.append((self.y, self.x))
        self.last_step_path = step_path
        self.path.append((self.y, self.x))
        self.daily_path.append((self.y, self.x))

    def scan(self, burned_true: np.ndarray, shared_knowledge: set[tuple[int, int]]) -> list[tuple[int, int]]:
        y0, x0 = int(self.y), int(self.x)
        radius = int(self.adaptive_vision)
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        mask = yy**2 + xx**2 <= radius**2

        y_min, y_max = max(0, y0 - radius), min(burned_true.shape[0], y0 + radius + 1)
        x_min, x_max = max(0, x0 - radius), min(burned_true.shape[1], x0 + radius + 1)
        region = burned_true[y_min:y_max, x_min:x_max]
        region_mask = mask[: region.shape[0], : region.shape[1]]
        new_observations = [
            (y_min + y, x_min + x)
            for y, x in np.argwhere(np.logical_and(region_mask, region > 0))
            if (y_min + y, x_min + x) not in shared_knowledge
        ]

        self.observed.update(new_observations)
        shared_knowledge.update(new_observations)
        fire_density = len(new_observations) / (np.pi * radius**2 + EPSILON)
        self._update_adaptive_vision(fire_density)
        return new_observations


def _dilate_binary_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    dilated = np.asarray(mask, dtype=bool)
    for _ in range(max(radius, 0)):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        neighborhoods = []
        for row_offset in range(3):
            for col_offset in range(3):
                neighborhoods.append(padded[row_offset : row_offset + dilated.shape[0], col_offset : col_offset + dilated.shape[1]])
        dilated = np.logical_or.reduce(neighborhoods)
    return dilated


def _select_spaced_targets(
    targets: list[tuple[int, int]],
    drone_positions: list[tuple[int, int]],
    max_targets: int,
    min_spacing_px: float,
) -> list[tuple[int, int]]:
    if not targets or max_targets <= 0:
        return []

    spacing = max(float(min_spacing_px), 0.0)
    scored_targets = sorted(
        targets,
        key=lambda target: min(np.hypot(target[0] - drone_y, target[1] - drone_x) for drone_y, drone_x in drone_positions)
        if drone_positions
        else 0.0,
    )
    selected: list[tuple[int, int]] = []
    for target_y, target_x in scored_targets:
        if all(np.hypot(target_y - selected_y, target_x - selected_x) >= spacing for selected_y, selected_x in selected):
            selected.append((target_y, target_x))
            if len(selected) >= max_targets:
                break

    if len(selected) < min(max_targets, len(scored_targets)):
        for target in scored_targets:
            if target not in selected:
                selected.append(target)
                if len(selected) >= max_targets:
                    break
    return selected


def _deduplicate_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        normalized = (int(cell[0]), int(cell[1]))
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _nearest_cell_to_point(cells: list[tuple[int, int]], point_y: float, point_x: float) -> tuple[int, int]:
    return min(cells, key=lambda cell: float(np.hypot(cell[0] - point_y, cell[1] - point_x)))


def _build_cluster_waypoints(
    cluster_cells: list[tuple[int, int]],
    growth_cells: list[tuple[int, int]],
    frontier_cells: list[tuple[int, int]],
    observed_footprint: set[tuple[int, int]],
    max_waypoints: int,
    prioritize_growth: bool = False,
) -> list[tuple[int, int]]:
    if not cluster_cells:
        return []

    growth_priority = [cell for cell in growth_cells if cell not in observed_footprint]
    uncovered_cells = [cell for cell in cluster_cells if cell not in observed_footprint]
    pool = growth_priority or uncovered_cells or growth_cells or cluster_cells
    centroid_y = float(np.mean([cell[0] for cell in pool]))
    centroid_x = float(np.mean([cell[1] for cell in pool]))
    centroid_cell = _nearest_cell_to_point(cluster_cells, centroid_y, centroid_x)

    waypoints = [centroid_cell]
    candidates = _deduplicate_cells(pool + cluster_cells)
    frontier_set = set(frontier_cells)
    growth_set = set(growth_cells)

    def _local_density_bonus(cell: tuple[int, int], cells: set[tuple[int, int]], radius: float = 6.0) -> float:
        if not cells:
            return 0.0
        return float(sum(1 for other in cells if np.hypot(cell[0] - other[0], cell[1] - other[1]) <= radius))

    def _waypoint_score(cell: tuple[int, int]) -> float:
        min_distance = min(float(np.hypot(cell[0] - waypoint[0], cell[1] - waypoint[1])) for waypoint in waypoints)
        growth_bonus = 7.5 if cell in growth_set else 0.0
        frontier_bonus = 3.0 if cell in frontier_set else 0.0
        unobserved_bonus = 2.5 if cell not in observed_footprint else 0.0
        covered_penalty = 2.0 if cell in observed_footprint else 0.0
        local_growth_bonus = 0.75 * _local_density_bonus(cell, growth_set)
        local_frontier_bonus = 0.15 * _local_density_bonus(cell, frontier_set)
        priority_multiplier = 1.35 if prioritize_growth and cell in growth_set else 1.0
        return priority_multiplier * (growth_bonus + frontier_bonus + unobserved_bonus + local_growth_bonus + local_frontier_bonus) + 0.08 * min_distance - covered_penalty

    while candidates and len(waypoints) < max(max_waypoints, 1):
        next_waypoint = max(candidates, key=_waypoint_score)
        if next_waypoint not in waypoints:
            waypoints.append(next_waypoint)
        candidates = [cell for cell in candidates if cell not in waypoints]

    return waypoints


def _build_target_clusters(
    growth_cells: list[tuple[int, int]],
    frontier_cells: list[tuple[int, int]],
    observed_footprint: set[tuple[int, int]],
    config: HybridMASConfig,
) -> list[TargetCluster]:
    grid_px = max(int(config.cluster_grid_px), 1)
    macro_grid_px = max(int(config.macro_sector_grid_px), grid_px)
    buckets: dict[tuple[int, int], dict[str, list[tuple[int, int]]]] = {}

    for cell in frontier_cells:
        key = (int(cell[0]) // grid_px, int(cell[1]) // grid_px)
        bucket = buckets.setdefault(key, {"growth": [], "frontier": []})
        bucket["frontier"].append((int(cell[0]), int(cell[1])))

    for cell in growth_cells:
        key = (int(cell[0]) // grid_px, int(cell[1]) // grid_px)
        bucket = buckets.setdefault(key, {"growth": [], "frontier": []})
        bucket["growth"].append((int(cell[0]), int(cell[1])))

    bucket_keys = set(buckets.keys())
    component_map: dict[tuple[int, int], int] = {}
    component_id = 0
    for start_key in sorted(bucket_keys):
        if start_key in component_map:
            continue
        stack = [start_key]
        component_map[start_key] = component_id
        while stack:
            row_key, col_key = stack.pop()
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    neighbor_key = (row_key + row_offset, col_key + col_offset)
                    if neighbor_key not in bucket_keys or neighbor_key in component_map:
                        continue
                    component_map[neighbor_key] = component_id
                    stack.append(neighbor_key)
        component_id += 1

    clusters: list[TargetCluster] = []
    prioritize_growth = config.target_strategy == "clustered_growth_frontier_priority"
    for cluster_index, (key, bucket) in enumerate(buckets.items()):
        cluster_cells = _deduplicate_cells(bucket["growth"] + bucket["frontier"])
        if not cluster_cells:
            continue
        growth_cluster_cells = _deduplicate_cells(bucket["growth"])
        frontier_cluster_cells = _deduplicate_cells(bucket["frontier"])
        waypoints = _build_cluster_waypoints(
            cluster_cells=cluster_cells,
            growth_cells=growth_cluster_cells,
            frontier_cells=frontier_cluster_cells,
            observed_footprint=observed_footprint,
            max_waypoints=config.max_waypoints_per_cluster,
            prioritize_growth=prioritize_growth,
        )
        centroid = waypoints[0] if waypoints else cluster_cells[0]
        uncovered_count = sum(1 for cell in cluster_cells if cell not in observed_footprint)
        score = float(len(growth_cluster_cells) * 3.0 + len(bucket["frontier"]) + uncovered_count * 0.5)
        source = "growth_frontier" if growth_cluster_cells and bucket["frontier"] else "growth" if growth_cluster_cells else "frontier"
        macro_sector_id = ((key[0] * grid_px) // macro_grid_px, (key[1] * grid_px) // macro_grid_px)
        clusters.append(
            TargetCluster(
                cluster_id=f"cluster_{key[0]}_{key[1]}_{cluster_index}",
                cells=cluster_cells,
                centroid=centroid,
                waypoints=waypoints,
                score=score,
                source=source,
                grid_key=key,
                macro_sector_id=macro_sector_id,
                component_id=component_map.get(key, -1),
            )
        )

    clusters.sort(key=lambda cluster: (-cluster.score, -len(cluster.cells), cluster.centroid[0], cluster.centroid[1]))
    return clusters


def _pairwise_distance_stats(points: list[tuple[int, int]]) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    distances: list[float] = []
    for point_index, first_point in enumerate(points[:-1]):
        for second_point in points[point_index + 1 :]:
            distances.append(float(np.hypot(first_point[0] - second_point[0], first_point[1] - second_point[1])))
    return float(np.mean(distances)), float(np.min(distances)) if distances else (0.0, 0.0)


def _select_diverse_clusters(clusters: list[TargetCluster], drone_count: int, config: HybridMASConfig) -> list[TargetCluster]:
    if drone_count <= 0 or not clusters:
        return []

    max_score = max((cluster.score for cluster in clusters), default=1.0)
    selected: list[TargetCluster] = []
    selected_sector_ids: set[tuple[int, int]] = set()
    selected_component_ids: set[int] = set()
    remaining = list(clusters)

    while remaining and len(selected) < drone_count:
        if not selected:
            best_cluster = max(remaining, key=lambda cluster: (cluster.score, len(cluster.cells)))
            selected.append(best_cluster)
            selected_sector_ids.add(best_cluster.macro_sector_id)
            selected_component_ids.add(best_cluster.component_id)
            remaining = [cluster for cluster in remaining if cluster.cluster_id != best_cluster.cluster_id]
            continue

        def _candidate_value(cluster: TargetCluster) -> float:
            normalized_score = cluster.score / max(max_score, EPSILON)
            min_distance = min(
                float(np.hypot(cluster.centroid[0] - selected_cluster.centroid[0], cluster.centroid[1] - selected_cluster.centroid[1]))
                for selected_cluster in selected
            )
            distance_term = min_distance / max(float(config.cluster_grid_px) * 4.0, EPSILON)
            new_sector_bonus = 1.0 if cluster.macro_sector_id not in selected_sector_ids else 0.0
            new_component_bonus = 1.0 if cluster.component_id not in selected_component_ids else 0.0
            same_sector_penalty = 1.0 if cluster.macro_sector_id in selected_sector_ids else 0.0
            return (
                config.diversity_alpha * normalized_score
                + config.diversity_beta * distance_term
                + config.diversity_gamma * new_sector_bonus
                + config.diversity_delta * new_component_bonus
                - config.diversity_same_sector_penalty * same_sector_penalty
            )

        best_cluster = max(remaining, key=lambda cluster: (_candidate_value(cluster), cluster.score, len(cluster.cells)))
        selected.append(best_cluster)
        selected_sector_ids.add(best_cluster.macro_sector_id)
        selected_component_ids.add(best_cluster.component_id)
        remaining = [cluster for cluster in remaining if cluster.cluster_id != best_cluster.cluster_id]

    return selected


def _sample_segment_cells(start: tuple[float, float], end: tuple[float, float]) -> list[tuple[int, int]]:
    dy = end[0] - start[0]
    dx = end[1] - start[1]
    steps = max(int(np.ceil(np.hypot(dy, dx))), 1)
    return [
        (int(round(start[0] + (dy * step_index) / steps)), int(round(start[1] + (dx * step_index) / steps)))
        for step_index in range(steps + 1)
    ]


def _scan_observation_area(
    drone: DroneAdvanced,
    burned_true: np.ndarray,
    observed_footprint: set[tuple[int, int]],
    observed_burned: set[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    radius = int(drone.adaptive_vision)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = yy**2 + xx**2 <= radius**2

    corridor_centers: list[tuple[int, int]] = []
    step_path = drone.last_step_path if len(drone.last_step_path) >= 2 else [(drone.y, drone.x)]
    if len(step_path) == 1:
        corridor_centers.append((int(round(step_path[0][0])), int(round(step_path[0][1]))))
    else:
        for start, end in zip(step_path[:-1], step_path[1:], strict=True):
            corridor_centers.extend(_sample_segment_cells(start, end))

    footprint_cells_set: set[tuple[int, int]] = set()
    for center_y, center_x in _deduplicate_cells(corridor_centers):
        y_min, y_max = max(0, center_y - radius), min(burned_true.shape[0], center_y + radius + 1)
        x_min, x_max = max(0, center_x - radius), min(burned_true.shape[1], center_x + radius + 1)
        region = burned_true[y_min:y_max, x_min:x_max]
        region_mask = mask[: region.shape[0], : region.shape[1]]
        for y, x in np.argwhere(region_mask):
            global_y = y_min + y
            global_x = x_min + x
            if (global_y, global_x) in observed_footprint:
                continue
            footprint_cells_set.add((global_y, global_x))

    footprint_cells = list(footprint_cells_set)

    burned_cells = [cell for cell in footprint_cells if burned_true[cell[0], cell[1]] > 0]
    clear_cells = [cell for cell in footprint_cells if burned_true[cell[0], cell[1]] == 0]

    drone.observed.update(burned_cells)
    observed_footprint.update(footprint_cells)
    observed_burned.update(burned_cells)
    fire_density = len(burned_cells) / (np.pi * radius**2 + EPSILON)
    drone._update_adaptive_vision(fire_density)
    return {"burned": burned_cells, "clear": clear_cells, "footprint": footprint_cells}


def _clone_weather_history(weather_data: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        date_str: {key: np.asarray(value, dtype=float).copy() for key, value in daily.items()}
        for date_str, daily in weather_data.items()
    }


def run_hybrid_ca_mas_optimized(
    static_data: dict[str, np.ndarray],
    burned_series: list[np.ndarray],
    burned_dates: list[str],
    fire_ca: FireCA,
    sensor_weather: dict[str, dict[str, np.ndarray]] | None = None,
    config: HybridMASConfig | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    hybrid_config = config or HybridMASConfig()
    rng = np.random.default_rng(seed)

    shared_burned_knowledge: set[tuple[int, int]] = set()
    global_observed_footprint: set[tuple[int, int]] = set()
    logs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    assignment_logs: list[dict[str, object]] = []
    colors = ["cyan", "magenta", "lime", "yellow", "orange"]
    total_observed_map = {f"Drone_{index + 1}": 0 for index in range(hybrid_config.drone_count)}
    optimizer = TrajectoryOptimizer()
    front_predictor = FireFrontPredictor(fire_ca)
    fire_ca_baseline = FireCA(static_data, _clone_weather_history(fire_ca.weather), config=fire_ca.config)
    fire_pred_hybrid = burned_series[0].copy()
    fire_pred_ca = burned_series[0].copy()

    initial_hotspots = np.argwhere(burned_series[0] > 0)
    if len(initial_hotspots) > 0:
        chosen = initial_hotspots[rng.choice(len(initial_hotspots), size=min(hybrid_config.drone_count, len(initial_hotspots)), replace=False)]
        drones = [
            DroneAdvanced(
                f"Drone_{index + 1}",
                int(y),
                int(x),
                vision=hybrid_config.sensing_radius_px,
                move=hybrid_config.speed_px_per_step,
                color=colors[index % len(colors)],
            )
            for index, (y, x) in enumerate(chosen)
        ]
    else:
        map_shape = burned_series[0].shape
        drones = [
            DroneAdvanced(
                f"Drone_{index + 1}",
                int(rng.integers(0, map_shape[0])),
                int(rng.integers(0, map_shape[1])),
                vision=hybrid_config.sensing_radius_px,
                move=hybrid_config.speed_px_per_step,
                color=colors[index % len(colors)],
            )
            for index in range(hybrid_config.drone_count)
        ]

    optimization_stats = {"total_assignments": 0}
    for day_index in range(1, len(burned_dates)):
        date_prev = burned_dates[day_index - 1]
        date_curr = burned_dates[day_index]
        burned_prev_true = burned_series[day_index - 1]
        burned_true = burned_series[day_index]
        sensor_daily_weather = (sensor_weather or fire_ca.weather).get(date_curr, {})
        daily_observed_footprint: set[tuple[int, int]] = set()
        daily_clear_observations: set[tuple[int, int]] = set()
        daily_burned_observations: set[tuple[int, int]] = set()
        weather = fire_ca.weather.get(date_curr, {})
        wind_speed = np.asarray(weather.get("wind_speed", np.zeros_like(static_data["dem"])), dtype=float)
        wind_dir = np.asarray(weather.get("wind_dir", np.zeros_like(static_data["dem"])), dtype=float)
        observed_before_assignment = set(global_observed_footprint)
        movement_memory = set(global_observed_footprint)
        forecast_seed = int(rng.integers(0, 2**32 - 1))
        forecast_candidate = fire_ca.spread_step(fire_pred_hybrid, date_prev, rng=np.random.default_rng(forecast_seed))
        forecast_ca = fire_ca_baseline.spread_step(fire_pred_ca, date_prev, rng=np.random.default_rng(forecast_seed))
        forecast_candidate_binary = np.asarray(forecast_candidate) > 0
        forecast_ca_binary = np.asarray(forecast_ca) > 0
        fire_pred_hybrid_binary = np.asarray(fire_pred_hybrid) > 0
        raw_predicted_growth_mask = np.logical_and(forecast_candidate_binary, np.logical_not(fire_pred_hybrid_binary))
        raw_predicted_growth_pixels = _count_binary_cells(raw_predicted_growth_mask)
        predicted_growth = [
            (int(y), int(x))
            for y, x in np.argwhere(raw_predicted_growth_mask)
            if (int(y), int(x)) not in shared_burned_knowledge and (int(y), int(x)) not in global_observed_footprint
        ]
        frontier_cells = [
            (int(y), int(x))
            for y, x in front_predictor.identify_fire_front(fire_pred_hybrid, buffer=1)
            if (int(y), int(x)) not in shared_burned_knowledge and (int(y), int(x)) not in global_observed_footprint
        ]
        target_guidance_map = np.zeros_like(forecast_candidate, dtype=np.uint8)
        for drone in drones:
            drone.reset_daily_path()
            drone.clear_assignment()

        cluster_count = 0
        assigned_cluster_count = 0
        selected_cluster_mean_pairwise_distance = 0.0
        selected_cluster_min_pairwise_distance = 0.0
        selected_macro_sector_count = 0
        selected_connected_component_count = 0
        drone_spatial_dispersion = 0.0
        mean_cluster_size_pixels = 0.0
        waypoints_total = 0
        mean_waypoints_per_drone = 0.0
        observed_cluster_pixels = 0
        observed_cluster_coverage_ratio = 0.0
        selected_target_pixels = 0
        assigned_target_pixels = 0
        assigned_cluster_cells: set[tuple[int, int]] = set()
        assigned_waypoint_cells: list[tuple[int, int]] = []

        if hybrid_config.target_strategy == "point_growth_frontier":
            if predicted_growth:
                targets = predicted_growth
                targeting_mode = "growth"
            else:
                targets = frontier_cells
                targeting_mode = "frontier"

            for target_y, target_x in targets:
                target_guidance_map[target_y, target_x] = 1

            drone_positions = [(int(drone.y), int(drone.x)) for drone in drones]
            if targets:
                spaced_targets = _select_spaced_targets(
                    targets=targets,
                    drone_positions=drone_positions,
                    max_targets=len(drones),
                    min_spacing_px=max(hybrid_config.min_target_spacing_px, hybrid_config.sensing_radius_px * 2.0),
                )
                selected_target_pixels = len(spaced_targets)
                target_guidance_map = np.zeros_like(forecast_candidate, dtype=np.uint8)
                for target_y, target_x in spaced_targets:
                    target_guidance_map[target_y, target_x] = 1

                assignments = optimizer.optimize_assignment(drone_positions, spaced_targets)
                assignment_distances: list[float] = []
                for drone_index, target_index in assignments:
                    if target_index < len(spaced_targets):
                        drones[drone_index].set_target(spaced_targets[target_index])
                        assignment_distances.append(float(np.hypot(drone_positions[drone_index][0] - spaced_targets[target_index][0], drone_positions[drone_index][1] - spaced_targets[target_index][1])))
                        assigned_target_pixels += 1
                        assigned_waypoint_cells.append(spaced_targets[target_index])
                if assignment_distances:
                    assignment_logs.append(
                        {
                            "date": date_curr,
                            "total_distance": float(sum(assignment_distances)),
                            "mean_distance": float(np.mean(assignment_distances)),
                            "num_pairs": len(assignment_distances),
                        }
                    )
                    optimization_stats["total_assignments"] += len(assignment_distances)
        else:
            targeting_mode = hybrid_config.target_strategy
            if hybrid_config.target_strategy == "clustered_frontier":
                cluster_growth_cells: list[tuple[int, int]] = []
            else:
                cluster_growth_cells = predicted_growth
            clusters = _build_target_clusters(
                growth_cells=cluster_growth_cells,
                frontier_cells=frontier_cells,
                observed_footprint=observed_before_assignment,
                config=hybrid_config,
            )
            cluster_count = len(clusters)
            mean_cluster_size_pixels = float(np.mean([len(cluster.cells) for cluster in clusters])) if clusters else 0.0
            drone_positions = [(int(drone.y), int(drone.x)) for drone in drones]
            if hybrid_config.target_strategy == "clustered_growth_frontier_diverse":
                selected_clusters = _select_diverse_clusters(clusters, len(drones), hybrid_config)
            else:
                selected_clusters = clusters[: min(len(drones), len(clusters))]
            selected_cluster_mean_pairwise_distance, selected_cluster_min_pairwise_distance = _pairwise_distance_stats([cluster.centroid for cluster in selected_clusters])
            selected_macro_sector_count = len({cluster.macro_sector_id for cluster in selected_clusters})
            selected_connected_component_count = len({cluster.component_id for cluster in selected_clusters})
            selected_target_pixels = int(sum(len(cluster.waypoints) for cluster in selected_clusters))
            for cluster in selected_clusters:
                for cell_y, cell_x in cluster.cells:
                    target_guidance_map[cell_y, cell_x] = 1

            cluster_targets = [cluster.centroid for cluster in selected_clusters]
            assignments = optimizer.optimize_assignment(drone_positions, cluster_targets)
            assignment_distances = []
            for drone_index, cluster_index in assignments:
                if cluster_index < len(selected_clusters):
                    cluster = selected_clusters[cluster_index]
                    drones[drone_index].set_cluster_patrol(cluster.cells, cluster.waypoints)
                    assigned_cluster_count += 1
                    assigned_target_pixels += len(cluster.waypoints)
                    waypoints_total += len(cluster.waypoints)
                    assigned_cluster_cells.update(cluster.cells)
                    assigned_waypoint_cells.extend(cluster.waypoints)
                    assignment_distances.append(float(np.hypot(drone_positions[drone_index][0] - cluster.centroid[0], drone_positions[drone_index][1] - cluster.centroid[1])))
            mean_waypoints_per_drone = waypoints_total / max(assigned_cluster_count, 1)
            if assignment_distances:
                assignment_logs.append(
                    {
                        "date": date_curr,
                        "total_distance": float(sum(assignment_distances)),
                        "mean_distance": float(np.mean(assignment_distances)),
                        "num_pairs": len(assignment_distances),
                    }
                )
                optimization_stats["total_assignments"] += len(assignment_distances)

        for _ in range(hybrid_config.steps_per_day):
            for drone in drones:
                drone.move_optimized(target_guidance_map, movement_memory, wind_speed, wind_dir, rng=rng)
                observations = _scan_observation_area(
                    drone,
                    burned_true,
                    daily_observed_footprint,
                    shared_burned_knowledge,
                )
                daily_burned_observations.update(observations["burned"])
                daily_clear_observations.update(observations["clear"])
                movement_memory.update(observations["footprint"])
                global_observed_footprint.update(observations["footprint"])
                total_observed_map[drone.id] += len(observations["burned"])
                logs.append(
                    {
                        "date": date_curr,
                        "drone": drone.id,
                        "newly_observed": len(observations["burned"]),
                        "total_observed": total_observed_map[drone.id],
                        "shared_cells": len(shared_burned_knowledge),
                    }
                )

        drone_spatial_dispersion, _ = _pairwise_distance_stats([(int(drone.y), int(drone.x)) for drone in drones])

        fire_pred_corrected = forecast_candidate.copy()
        for y, x in shared_burned_knowledge:
            fire_pred_corrected[y, x] = 1 if burned_true[y, x] > 0 else 0
        for y, x in daily_clear_observations:
            fire_pred_corrected[y, x] = 0

        burned_mask = np.zeros_like(forecast_candidate, dtype=bool)
        for y, x in daily_burned_observations:
            burned_mask[y, x] = True
        clear_mask = np.zeros_like(forecast_candidate, dtype=bool)
        for y, x in daily_clear_observations:
            clear_mask[y, x] = True

        reinforced_zone = _dilate_binary_mask(burned_mask, radius=hybrid_config.burned_assimilation_radius_px)
        suppressed_zone = _dilate_binary_mask(clear_mask, radius=hybrid_config.clear_assimilation_radius_px)

        reinforced_front = np.logical_and(reinforced_zone, forecast_candidate > 0)
        reinforced_growth = np.logical_and(reinforced_zone, _dilate_binary_mask(fire_pred_hybrid > 0, radius=1))
        suppressed_front = np.logical_and(suppressed_zone, np.logical_and(forecast_candidate > 0, fire_pred_hybrid == 0))

        fire_pred_corrected[reinforced_front] = 1
        fire_pred_corrected[reinforced_growth] = 1
        fire_pred_corrected[suppressed_front] = 0
        fire_pred_corrected[np.logical_and(suppressed_zone, np.logical_not(reinforced_zone))] = 0

        fire_pred_corrected_binary = np.asarray(fire_pred_corrected) > 0
        observed_footprint_mask = np.zeros_like(forecast_candidate, dtype=bool)
        for y, x in daily_observed_footprint:
            observed_footprint_mask[y, x] = True

        assimilated_weather = _assimilate_local_weather(
            prior_weather=fire_ca.weather.get(date_curr, {}),
            sensor_weather=sensor_daily_weather,
            observed_cells=daily_observed_footprint,
            blend_weight=hybrid_config.weather_blend_weight,
            radius=hybrid_config.weather_assimilation_radius_px,
        )
        fire_ca.weather[date_curr] = assimilated_weather

        fire_pred_without_drones = forecast_ca.copy()
        fire_pred_ca_binary = np.asarray(fire_pred_ca) > 0

        hybrid_metrics = compute_fire_metrics(fire_pred_corrected, burned_true)
        ca_metrics = compute_fire_metrics(fire_pred_without_drones, burned_true)
        frontier_mask = build_frontier_evaluation_mask(burned_prev_true, burned_true)
        growth_mask = build_daily_growth_mask(burned_prev_true, burned_true)
        frontier_hybrid_metrics = compute_fire_metrics(fire_pred_corrected, burned_true, evaluation_mask=frontier_mask)
        frontier_ca_metrics = compute_fire_metrics(fire_pred_without_drones, burned_true, evaluation_mask=frontier_mask)
        growth_hybrid_metrics = compute_fire_metrics(fire_pred_corrected, burned_true, evaluation_mask=growth_mask)
        growth_ca_metrics = compute_fire_metrics(fire_pred_without_drones, burned_true, evaluation_mask=growth_mask)
        predicted_growth_hybrid_mask = np.logical_and(fire_pred_corrected_binary, np.logical_not(fire_pred_hybrid_binary))
        predicted_growth_ca_mask = np.logical_and(forecast_ca_binary, np.logical_not(fire_pred_ca_binary))
        corrected_to_burned_mask = np.logical_and(fire_pred_corrected_binary, np.logical_not(forecast_candidate_binary))
        corrected_to_clear_mask = np.logical_and(np.logical_not(fire_pred_corrected_binary), forecast_candidate_binary)
        corrected_any_mask = np.logical_xor(fire_pred_corrected_binary, forecast_candidate_binary)
        observed_frontier_pixels = _count_binary_cells(np.logical_and(observed_footprint_mask, frontier_mask))
        frontier_pixels = _count_binary_cells(frontier_mask)
        observed_growth_pixels = _count_binary_cells(np.logical_and(observed_footprint_mask, growth_mask))
        if hybrid_config.use_spread_bias:
            observed_burned_mask = np.zeros_like(forecast_candidate, dtype=bool)
            for y, x in daily_burned_observations:
                observed_burned_mask[y, x] = True
            observed_growth_mask = np.logical_and(observed_burned_mask, growth_mask)
            observed_frontier_burned_mask = np.logical_and(observed_burned_mask, frontier_mask)
            observed_clear_growth_mask = np.logical_and(clear_mask, np.logical_and(forecast_candidate > 0, fire_pred_hybrid == 0))
            next_day_bias = _build_next_day_spread_bias(
                reference_shape=forecast_candidate.shape,
                observed_growth_mask=observed_growth_mask,
                observed_frontier_burned_mask=observed_frontier_burned_mask,
                observed_clear_growth_mask=observed_clear_growth_mask,
                config=hybrid_config,
            )
            fire_ca.set_spread_bias(date_curr, next_day_bias)
        if assigned_cluster_cells:
            assigned_cluster_mask = np.zeros_like(forecast_candidate, dtype=bool)
            for y, x in assigned_cluster_cells:
                assigned_cluster_mask[y, x] = True
            observed_cluster_pixels = _count_binary_cells(np.logical_and(observed_footprint_mask, assigned_cluster_mask))
            observed_cluster_coverage_ratio = observed_cluster_pixels / (len(assigned_cluster_cells) + EPSILON)
        unique_assigned_waypoints = _deduplicate_cells(assigned_waypoint_cells)
        predicted_growth_set = set(predicted_growth)
        frontier_cell_set = set(frontier_cells)
        growth_waypoint_count = sum(1 for cell in unique_assigned_waypoints if cell in predicted_growth_set)
        frontier_waypoint_count = sum(1 for cell in unique_assigned_waypoints if cell in frontier_cell_set)
        unobserved_waypoint_count = sum(1 for cell in unique_assigned_waypoints if cell not in observed_before_assignment)
        waypoint_denominator = max(len(unique_assigned_waypoints), 1)
        daily_metrics = HybridDayMetrics(
            date=date_curr,
            targeting_mode=targeting_mode,
            cluster_count=cluster_count,
            assigned_cluster_count=assigned_cluster_count,
            selected_cluster_mean_pairwise_distance=selected_cluster_mean_pairwise_distance,
            selected_cluster_min_pairwise_distance=selected_cluster_min_pairwise_distance,
            selected_macro_sector_count=selected_macro_sector_count,
            selected_connected_component_count=selected_connected_component_count,
            drone_spatial_dispersion=drone_spatial_dispersion,
            mean_cluster_size_pixels=mean_cluster_size_pixels,
            waypoints_total=waypoints_total,
            mean_waypoints_per_drone=mean_waypoints_per_drone,
            observed_cluster_pixels=observed_cluster_pixels,
            observed_cluster_coverage_ratio=observed_cluster_coverage_ratio,
            growth_waypoint_ratio=growth_waypoint_count / waypoint_denominator,
            frontier_waypoint_ratio=frontier_waypoint_count / waypoint_denominator,
            unobserved_waypoint_ratio=unobserved_waypoint_count / waypoint_denominator,
            observed_growth_pixels=observed_growth_pixels,
            growth_coverage_ratio=observed_growth_pixels / (_count_binary_cells(growth_mask) + EPSILON),
            iou_hybrid=hybrid_metrics.iou,
            iou_ca=ca_metrics.iou,
            iou_improvement=hybrid_metrics.iou - ca_metrics.iou,
            frontier_iou_hybrid=frontier_hybrid_metrics.iou,
            frontier_iou_ca=frontier_ca_metrics.iou,
            frontier_iou_improvement=frontier_hybrid_metrics.iou - frontier_ca_metrics.iou,
            dice=hybrid_metrics.dice,
            precision_hybrid=hybrid_metrics.precision,
            recall_hybrid=hybrid_metrics.recall,
            f1_hybrid=hybrid_metrics.f1,
            precision_ca=ca_metrics.precision,
            recall_ca=ca_metrics.recall,
            f1_ca=ca_metrics.f1,
            frontier_f1_hybrid=frontier_hybrid_metrics.f1,
            frontier_f1_ca=frontier_ca_metrics.f1,
            frontier_f1_improvement=frontier_hybrid_metrics.f1 - frontier_ca_metrics.f1,
            precision_improvement=hybrid_metrics.precision - ca_metrics.precision,
            recall_improvement=hybrid_metrics.recall - ca_metrics.recall,
            f1_improvement=hybrid_metrics.f1 - ca_metrics.f1,
            growth_iou_hybrid=growth_hybrid_metrics.iou,
            growth_iou_ca=growth_ca_metrics.iou,
            growth_iou_improvement=growth_hybrid_metrics.iou - growth_ca_metrics.iou,
            growth_f1_hybrid=growth_hybrid_metrics.f1,
            growth_f1_ca=growth_ca_metrics.f1,
            growth_f1_improvement=growth_hybrid_metrics.f1 - growth_ca_metrics.f1,
            true_growth_pixels=_count_binary_cells(growth_mask),
            predicted_growth_hybrid_pixels=_count_binary_cells(predicted_growth_hybrid_mask),
            predicted_growth_ca_pixels=_count_binary_cells(predicted_growth_ca_mask),
            raw_predicted_growth_pixels=raw_predicted_growth_pixels,
            filtered_growth_target_pixels=len(predicted_growth),
            frontier_candidate_pixels=len(frontier_cells),
            selected_target_pixels=selected_target_pixels,
            assigned_target_pixels=assigned_target_pixels,
            observed_footprint_pixels=len(daily_observed_footprint),
            observed_burned_pixels=len(daily_burned_observations),
            observed_clear_pixels=len(daily_clear_observations),
            observed_frontier_pixels=observed_frontier_pixels,
            frontier_pixels=frontier_pixels,
            frontier_coverage_ratio=observed_frontier_pixels / (frontier_pixels + EPSILON),
            corrected_to_burned_pixels=_count_binary_cells(corrected_to_burned_mask),
            corrected_to_clear_pixels=_count_binary_cells(corrected_to_clear_mask),
            corrected_total_pixels=_count_binary_cells(corrected_any_mask),
        )
        fire_pred_hybrid = fire_pred_corrected.copy()
        fire_pred_ca = forecast_ca.copy()
        results.append(
            {
                "date": date_curr,
                "corrected_fire": fire_pred_corrected.copy(),
                "ca_prediction": fire_pred_without_drones.copy(),
                "drone_states": [
                    {
                        "id": drone.id,
                        "y": drone.y,
                        "x": drone.x,
                        "color": drone.color,
                        "daily_path": list(drone.daily_path),
                    }
                    for drone in drones
                ],
                "metrics": daily_metrics,
            }
        )

    return {
        "logs": logs,
        "results": results,
        "metrics": [result["metrics"] for result in results],
        "assign_logs": assignment_logs,
        "optimization_stats": optimization_stats,
    }


def summarize_hybrid_metrics(metrics: list[HybridDayMetrics]) -> dict[str, float]:
    if not metrics:
        return {
            "mean_cluster_count": 0.0,
            "mean_assigned_cluster_count": 0.0,
            "mean_selected_cluster_mean_pairwise_distance": 0.0,
            "mean_selected_cluster_min_pairwise_distance": 0.0,
            "mean_selected_macro_sector_count": 0.0,
            "mean_selected_connected_component_count": 0.0,
            "mean_drone_spatial_dispersion": 0.0,
            "mean_cluster_size_pixels": 0.0,
            "mean_waypoints_total": 0.0,
            "mean_waypoints_per_drone": 0.0,
            "mean_observed_cluster_pixels": 0.0,
            "mean_observed_cluster_coverage_ratio": 0.0,
            "mean_growth_waypoint_ratio": 0.0,
            "mean_frontier_waypoint_ratio": 0.0,
            "mean_unobserved_waypoint_ratio": 0.0,
            "mean_observed_growth_pixels": 0.0,
            "mean_growth_coverage_ratio": 0.0,
            "mean_iou_hybrid": 0.0,
            "mean_iou_ca": 0.0,
            "mean_iou_improvement": 0.0,
            "mean_frontier_iou_hybrid": 0.0,
            "mean_frontier_iou_ca": 0.0,
            "mean_frontier_iou_improvement": 0.0,
            "mean_f1_hybrid": 0.0,
            "mean_f1_ca": 0.0,
            "mean_f1_improvement": 0.0,
            "mean_frontier_f1_hybrid": 0.0,
            "mean_frontier_f1_ca": 0.0,
            "mean_frontier_f1_improvement": 0.0,
        }
    return {
        "mean_cluster_count": float(np.mean([metric.cluster_count for metric in metrics])),
        "mean_assigned_cluster_count": float(np.mean([metric.assigned_cluster_count for metric in metrics])),
        "mean_selected_cluster_mean_pairwise_distance": float(np.mean([metric.selected_cluster_mean_pairwise_distance for metric in metrics])),
        "mean_selected_cluster_min_pairwise_distance": float(np.mean([metric.selected_cluster_min_pairwise_distance for metric in metrics])),
        "mean_selected_macro_sector_count": float(np.mean([metric.selected_macro_sector_count for metric in metrics])),
        "mean_selected_connected_component_count": float(np.mean([metric.selected_connected_component_count for metric in metrics])),
        "mean_drone_spatial_dispersion": float(np.mean([metric.drone_spatial_dispersion for metric in metrics])),
        "mean_cluster_size_pixels": float(np.mean([metric.mean_cluster_size_pixels for metric in metrics])),
        "mean_waypoints_total": float(np.mean([metric.waypoints_total for metric in metrics])),
        "mean_waypoints_per_drone": float(np.mean([metric.mean_waypoints_per_drone for metric in metrics])),
        "mean_observed_cluster_pixels": float(np.mean([metric.observed_cluster_pixels for metric in metrics])),
        "mean_observed_cluster_coverage_ratio": float(np.mean([metric.observed_cluster_coverage_ratio for metric in metrics])),
        "mean_growth_waypoint_ratio": float(np.mean([metric.growth_waypoint_ratio for metric in metrics])),
        "mean_frontier_waypoint_ratio": float(np.mean([metric.frontier_waypoint_ratio for metric in metrics])),
        "mean_unobserved_waypoint_ratio": float(np.mean([metric.unobserved_waypoint_ratio for metric in metrics])),
        "mean_observed_growth_pixels": float(np.mean([metric.observed_growth_pixels for metric in metrics])),
        "mean_growth_coverage_ratio": float(np.mean([metric.growth_coverage_ratio for metric in metrics])),
        "mean_iou_hybrid": float(np.mean([metric.iou_hybrid for metric in metrics])),
        "mean_iou_ca": float(np.mean([metric.iou_ca for metric in metrics])),
        "mean_iou_improvement": float(np.mean([metric.iou_improvement for metric in metrics])),
        "mean_frontier_iou_hybrid": float(np.mean([metric.frontier_iou_hybrid for metric in metrics])),
        "mean_frontier_iou_ca": float(np.mean([metric.frontier_iou_ca for metric in metrics])),
        "mean_frontier_iou_improvement": float(np.mean([metric.frontier_iou_improvement for metric in metrics])),
        "mean_f1_hybrid": float(np.mean([metric.f1_hybrid for metric in metrics])),
        "mean_f1_ca": float(np.mean([metric.f1_ca for metric in metrics])),
        "mean_f1_improvement": float(np.mean([metric.f1_improvement for metric in metrics])),
        "mean_frontier_f1_hybrid": float(np.mean([metric.frontier_f1_hybrid for metric in metrics])),
        "mean_frontier_f1_ca": float(np.mean([metric.frontier_f1_ca for metric in metrics])),
        "mean_frontier_f1_improvement": float(np.mean([metric.frontier_f1_improvement for metric in metrics])),
        "mean_growth_iou_hybrid": float(np.mean([metric.growth_iou_hybrid for metric in metrics])),
        "mean_growth_iou_ca": float(np.mean([metric.growth_iou_ca for metric in metrics])),
        "mean_growth_iou_improvement": float(np.mean([metric.growth_iou_improvement for metric in metrics])),
        "mean_growth_f1_hybrid": float(np.mean([metric.growth_f1_hybrid for metric in metrics])),
        "mean_growth_f1_ca": float(np.mean([metric.growth_f1_ca for metric in metrics])),
        "mean_growth_f1_improvement": float(np.mean([metric.growth_f1_improvement for metric in metrics])),
        "mean_raw_predicted_growth_pixels": float(np.mean([metric.raw_predicted_growth_pixels for metric in metrics])),
        "mean_filtered_growth_target_pixels": float(np.mean([metric.filtered_growth_target_pixels for metric in metrics])),
        "mean_frontier_candidate_pixels": float(np.mean([metric.frontier_candidate_pixels for metric in metrics])),
        "mean_selected_target_pixels": float(np.mean([metric.selected_target_pixels for metric in metrics])),
        "mean_assigned_target_pixels": float(np.mean([metric.assigned_target_pixels for metric in metrics])),
        "mean_frontier_coverage_ratio": float(np.mean([metric.frontier_coverage_ratio for metric in metrics])),
        "mean_observed_footprint_pixels": float(np.mean([metric.observed_footprint_pixels for metric in metrics])),
        "mean_observed_burned_pixels": float(np.mean([metric.observed_burned_pixels for metric in metrics])),
        "mean_observed_clear_pixels": float(np.mean([metric.observed_clear_pixels for metric in metrics])),
        "mean_corrected_to_burned_pixels": float(np.mean([metric.corrected_to_burned_pixels for metric in metrics])),
        "mean_corrected_to_clear_pixels": float(np.mean([metric.corrected_to_clear_pixels for metric in metrics])),
        "mean_corrected_total_pixels": float(np.mean([metric.corrected_total_pixels for metric in metrics])),
    }


def hybrid_metrics_to_dicts(metrics: list[HybridDayMetrics]) -> list[dict[str, object]]:
    return [asdict(metric) for metric in metrics]


def run_ca_baseline(
    project_root: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int = 42,
    use_barriers: bool = False,
    use_viirs_init: bool = False,
    viirs_history_days: int | None = None,
) -> dict[str, object]:
    dataset = load_project_data(project_root)
    burned_dates = dataset["burned_dates"]
    burned_series = dataset["burned_series"]
    viirs_series = dataset["viirs_series"]
    static_arrays = dataset["static_arrays"]
    weather_data = dataset["weather_data"]
    prior_weather, _ = _build_weather_prior(weather_data, static_arrays["dem"])

    start_date = start_date or burned_dates[0]
    end_date = end_date or burned_dates[min(len(burned_dates) - 1, 3)]
    if start_date not in burned_dates or end_date not in burned_dates:
        raise ValueError("start-date and end-date must exist in burned series")
    start_idx = burned_dates.index(start_date)
    end_idx = burned_dates.index(end_date)
    if start_idx >= end_idx:
        raise ValueError("start-date must be earlier than end-date")

    model = FireCA(static_arrays, prior_weather, config=FireCAConfig(use_barriers=use_barriers))
    rollout_dates = burned_dates[start_idx:end_idx]
    initial_fire = burned_series[start_idx]
    if use_viirs_init:
        viirs_initial = build_initial_fire_from_viirs_history(viirs_series, start_idx, history_days=viirs_history_days)
        if int(viirs_initial.sum()) > 0:
            initial_fire = viirs_initial
    predictions = model.rollout(initial_fire, rollout_dates, seed=seed)
    metrics = evaluate_rollout(
        predicted_series=predictions[1:],
        true_series=burned_series[start_idx + 1 : end_idx + 1],
        evaluation_dates=burned_dates[start_idx + 1 : end_idx + 1],
    )
    return {
        "dataset": dataset,
        "model": model,
        "start_date": start_date,
        "end_date": end_date,
        "seed": seed,
        "use_barriers": use_barriers,
        "use_viirs_init": use_viirs_init,
        "viirs_history_days": viirs_history_days,
        "predictions": predictions,
        "metrics": metrics,
        "summary": summarize_metrics(metrics),
    }


def run_hybrid_experiment(
    project_root: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int = 42,
    use_barriers: bool = False,
    drone_count: int = 10,
    vision: float = 8.0,
    move: float = 35.0,
    steps_per_day: int = 24,
    target_strategy: str = "clustered_growth_frontier",
    use_spread_bias: bool = False,
    use_viirs_init: bool = False,
    viirs_history_days: int | None = None,
) -> dict[str, object]:
    dataset = load_project_data(project_root)
    burned_dates = dataset["burned_dates"]
    burned_series = dataset["burned_series"]
    viirs_series = dataset["viirs_series"]
    static_arrays = dataset["static_arrays"]
    weather_data = dataset["weather_data"]
    prior_weather, sensor_weather = _build_weather_prior(weather_data, static_arrays["dem"])

    start_date = start_date or burned_dates[0]
    end_date = end_date or burned_dates[min(len(burned_dates) - 1, 4)]
    if start_date not in burned_dates or end_date not in burned_dates:
        raise ValueError("start-date and end-date must exist in burned series")
    start_idx = burned_dates.index(start_date)
    end_idx = burned_dates.index(end_date)
    if start_idx >= end_idx:
        raise ValueError("start-date must be earlier than end-date")

    initial_fire = burned_series[start_idx]
    if use_viirs_init:
        viirs_initial = build_initial_fire_from_viirs_history(viirs_series, start_idx, history_days=viirs_history_days)
        if int(viirs_initial.sum()) > 0:
            initial_fire = viirs_initial

    fire_ca = FireCA(static_arrays, prior_weather, config=FireCAConfig(use_barriers=use_barriers))
    hybrid_result = run_hybrid_ca_mas_optimized(
        static_data=static_arrays,
        burned_series=[initial_fire.copy(), *burned_series[start_idx + 1 : end_idx + 1]],
        burned_dates=burned_dates[start_idx : end_idx + 1],
        fire_ca=fire_ca,
        sensor_weather={date: sensor_weather[date] for date in burned_dates[start_idx : end_idx + 1]},
        config=HybridMASConfig(
            drone_count=drone_count,
            sensing_radius_px=vision,
            speed_px_per_step=move,
            steps_per_day=steps_per_day,
            target_strategy=target_strategy,
            use_spread_bias=use_spread_bias,
        ),
        seed=seed,
    )
    return {
        "dataset": dataset,
        "model": fire_ca,
        "start_date": start_date,
        "end_date": end_date,
        "seed": seed,
        "use_barriers": use_barriers,
        "use_viirs_init": use_viirs_init,
        "viirs_history_days": viirs_history_days,
        "drone_count": drone_count,
        "vision": vision,
        "move": move,
        "steps_per_day": steps_per_day,
        "target_strategy": target_strategy,
        "use_spread_bias": use_spread_bias,
        "hybrid_result": hybrid_result,
        "summary": summarize_hybrid_metrics(hybrid_result["metrics"]),
    }


def serialize_ca_run(run_result: dict[str, object]) -> dict[str, object]:
    return {
        "start_date": run_result["start_date"],
        "end_date": run_result["end_date"],
        "seed": run_result["seed"],
        "use_barriers": run_result["use_barriers"],
        "summary": run_result["summary"],
        "daily_metrics": [asdict(metric) for metric in run_result["metrics"]],
        "validation_findings": [asdict(finding) for finding in run_result["dataset"]["validation_findings"]],
    }


def serialize_hybrid_run(run_result: dict[str, object]) -> dict[str, object]:
    hybrid_result = run_result["hybrid_result"]
    return {
        "start_date": run_result["start_date"],
        "end_date": run_result["end_date"],
        "seed": run_result["seed"],
        "use_barriers": run_result["use_barriers"],
        "drone_count": run_result["drone_count"],
        "vision": run_result["vision"],
        "move": run_result["move"],
        "steps_per_day": run_result["steps_per_day"],
        "target_strategy": run_result["target_strategy"],
        "use_spread_bias": run_result["use_spread_bias"],
        "summary": run_result["summary"],
        "optimization_stats": hybrid_result["optimization_stats"],
        "daily_metrics": hybrid_metrics_to_dicts(hybrid_result["metrics"]),
        "assignment_logs": hybrid_result["assign_logs"],
        "validation_findings": [asdict(finding) for finding in run_result["dataset"]["validation_findings"]],
    }


def inspect_legacy_project(project_root: str | Path) -> dict[str, object]:
    layout = resolve_legacy_project_layout(project_root)
    summary = summarize_legacy_project_data(layout)
    findings = validate_static_layers(normalize_static_layers(load_static_layers(layout)))
    summary["validation_findings"] = [asdict(finding) for finding in findings]
    return summary


def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configs for the export pipeline.") from exc

    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {file_path}")
    return data


def build_export_paths(raw_data_dir: str | Path, case_name: str) -> ExportPaths:
    root = resolve_project_path(str(raw_data_dir))
    raw_case_dir = root / case_name
    static_dir = raw_case_dir / "static"
    time_series_dir = raw_case_dir / "time_series"
    burned_dir = time_series_dir / "burned_area"
    viirs_dir = time_series_dir / "fire_viirs"
    weather_dir = time_series_dir / "weather_daily"
    metadata_dir = raw_case_dir / "metadata"
    return ExportPaths(case_name, raw_case_dir, static_dir, time_series_dir, burned_dir, viirs_dir, weather_dir, metadata_dir)


def ensure_export_dirs(paths: ExportPaths) -> None:
    for directory in (paths.raw_case_dir, paths.static_dir, paths.time_series_dir, paths.burned_dir, paths.viirs_dir, paths.weather_dir, paths.metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)


def load_export_config(defaults_path: str | Path) -> GEEExportConfig:
    data = _load_yaml_file(resolve_project_path(str(defaults_path)))
    study = data.get("study", {})
    paths = data.get("paths", {})
    gee = data.get("gee", {})
    return GEEExportConfig(
        project_id=str(gee.get("project_id", "")),
        raw_data_dir=str(paths.get("raw_data_dir", "outputs/raw")),
        crs=str(study.get("crs", "EPSG:4326")),
        grid_resolution_m=int(study.get("grid_resolution_m", 100)),
        start_date=str(study.get("start_date", "2021-07-01")),
        end_date=str(study.get("end_date", "2021-09-30")),
        aoi_backend=str(gee.get("aoi_backend", "geojson")),
        earth_engine_asset_id=gee.get("earth_engine_asset_id"),
        export_static_layers=bool(gee.get("export_static_layers", True)),
        export_time_series=bool(gee.get("export_time_series", True)),
    )


def load_cases(cases_path: str | Path) -> dict[str, CaseDefinition]:
    data = _load_yaml_file(resolve_project_path(str(cases_path)))
    cases = data.get("cases", {})
    loaded: dict[str, CaseDefinition] = {}
    if isinstance(cases, list):
        for case_data in cases:
            if not isinstance(case_data, dict):
                continue
            case_name = str(case_data.get("id", "")).strip()
            if not case_name:
                continue
            loaded[case_name] = CaseDefinition(
                name=case_name,
                aoi_source=str(case_data.get("aoi_source") or case_data.get("aoi_geojson") or ""),
                start_date=str(case_data.get("start_date", "")),
                end_date=str(case_data.get("end_date", "")),
                notes=str(case_data.get("notes") or case_data.get("timezone") or ""),
            )
        return loaded
    for case_name, case_data in cases.items():
        if not isinstance(case_data, dict):
            continue
        loaded[case_name] = CaseDefinition(
            name=case_name,
            aoi_source=str(case_data.get("aoi_source", "")),
            start_date=str(case_data.get("start_date", "")),
            end_date=str(case_data.get("end_date", "")),
            notes=str(case_data.get("notes", "")),
        )
    return loaded


def _import_ee_modules():
    try:
        import ee
        import geemap
    except ImportError as exc:
        raise RuntimeError("Google Earth Engine dependencies are missing. Install `earthengine-api` and `geemap`.") from exc
    return ee, geemap


def initialize_earth_engine(project_id: str):
    ee, _ = _import_ee_modules()
    if not project_id:
        raise ValueError("Missing GEE project_id in config.")
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    return ee


def load_aoi(case_def: CaseDefinition, config: GEEExportConfig):
    ee, geemap = _import_ee_modules()
    backend = config.aoi_backend.lower().strip()
    if backend == "asset":
        asset_id = config.earth_engine_asset_id or case_def.aoi_source
        if not asset_id:
            raise ValueError(f"No Earth Engine asset id configured for case {case_def.name}.")
        return ee.FeatureCollection(asset_id)

    aoi_path = resolve_project_path(case_def.aoi_source)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found for case {case_def.name}: {aoi_path}")
    if aoi_path.suffix.lower() in {".geojson", ".json"}:
        return geemap.geojson_to_ee(str(aoi_path))
    if aoi_path.suffix.lower() == ".shp":
        return geemap.shp_to_ee(str(aoi_path))
    raise ValueError(f"Unsupported AOI source format: {aoi_path.suffix}")


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_case_export(case_def: CaseDefinition, export_config: GEEExportConfig, dry_run: bool = False) -> dict[str, object]:
    export_paths = build_export_paths(export_config.raw_data_dir, case_def.name)
    ensure_export_dirs(export_paths)
    metadata = {
        "case_name": case_def.name,
        "aoi_source": case_def.aoi_source,
        "start_date": case_def.start_date or export_config.start_date,
        "end_date": case_def.end_date or export_config.end_date,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "dry_run": dry_run,
    }

    if dry_run:
        _write_metadata(export_paths.metadata_dir / "dry_run_export.json", metadata)
        return {
            "status": "dry_run",
            "case_name": case_def.name,
            "raw_case_dir": str(export_paths.raw_case_dir),
            "metadata_path": str(export_paths.metadata_dir / "dry_run_export.json"),
        }

    ee = initialize_earth_engine(export_config.project_id)
    aoi = load_aoi(case_def, export_config)
    _write_metadata(export_paths.metadata_dir / "export_request.json", metadata)

    return {
        "status": "initialized",
        "case_name": case_def.name,
        "raw_case_dir": str(export_paths.raw_case_dir),
        "aoi_type": type(aoi).__name__,
        "ee_initialized": ee is not None,
    }