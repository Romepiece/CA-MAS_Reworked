# Hybrid CA+MAS Wildfire Front Forecasting for the 2021 Dixie Fire

Retrospective wildfire-front forecasting with a stochastic cellular automaton, simulated UAV observations, and binary data assimilation.

Title: An adaptive hybrid model for wildfire front forecasting based on cellular automata, multi-agent UAV observations, and binary data assimilation: A case study of the 2021 Dixie Fire.

## Project Overview

This repository contains the code, final article-backed outputs, and LaTeX manuscript materials for a retrospective case-study evaluation of a hybrid wildfire forecasting framework applied to the 2021 Dixie Fire in California, USA.

The central idea of the project is to combine an interpretable cellular automaton wildfire-spread model with a multi-agent system of simulated UAV observers. The cellular automaton produces a daily wildfire-front forecast from terrain, vegetation, barrier, and weather layers. The UAV agents are then directed toward informative regions of predicted fire growth, and their observations are assimilated back into the forecast through a lightweight binary confirm-deny update before the next simulation step.

In the verified August evaluation window used in the manuscript, the hybrid CA+MAS configuration improved the mean IoU from 0.6664 for the standalone CA baseline to 0.6992, with the mean F1 score increasing from 0.7998 to 0.8229.

## Dataset Information

The study uses a prepared retrospective geospatial dataset for the Dixie Fire case study on a common 100 m grid. The full prepared stack spans 1 July 2021 to 30 September 2021, while the quantitative comparison reported in the manuscript is restricted to the verified 5-day mid-fire window from 14 to 18 August 2021, with 14 August used for initialization and daily forecast skill reported for 15 to 18 August.

**Reproducibility note:**
The dataset assembly scripts and configuration files were frequently modified during the project. As a result, the assembly files you see in the repository may differ from those used to generate the results in the published article. Therefore, to exactly reproduce the results reported in the scientific project, you should use the provided dataset located in the `1_project_code/data` folder.

The case-study stack combines the following data sources:

* DEM, slope, and aspect from SRTM-derived terrain layers.
* Land-cover baseline from ESA WorldCover.
* Vegetation condition from MODIS NDVI.
* Burned-area reference from MODIS MCD64A1.
* Active-fire guidance from VIIRS.
* Meteorological forcing from ERA5-Land.
* Road and water barriers derived from OpenStreetMap and QGIS processing.

Important note: this package does not include the full raw legacy raster dataset. The simulation stage in `1_project_code/01_simulation_app` consumes prepared rasters, while the dataset-generation stage in `1_project_code/02_dataset_generation` documents how upstream case-data assembly can be reproduced.

## Algorithm Overview

The workflow implemented in this package has three main layers:

* Cellular automaton forecast: the CA propagates the daily fire front using topography, fuel proxies, vegetation condition, meteorological variables, and optional barrier masks.
* Multi-agent UAV layer: simulated drones are assigned to predicted growth and frontier regions using configurable targeting strategies and assignment logic.
* Binary data assimilation: observed burned cells reinforce the forecast, observed clear cells suppress false positives, and the observed footprint can also support local blending of daily weather fields.

The verified manuscript configuration uses the following key settings:

* Grid resolution: 100 m.
* Grid size: 1115 x 1115 cells.
* Time step: 1 day.
* Base CA spread probability: 0.15.
* Number of drones: 5.
* UAV sensing radius: 8 px.
* UAV movement speed: 35 px per step.
* Intra-day UAV updates: 24 steps per simulated day.
* Assimilation mode: binary confirm-deny.

## Repository Structure

* `1_project_code/01_simulation_app` is Stage 1 and contains the simulation-side code: CLI, CA core, assimilation logic, reporting, and the simulation config template.
* `1_project_code/02_dataset_generation` is Stage 2 and contains the Earth Engine-side collection and preprocessing code, the bundled case configuration, and the AOI inputs.
* `1_project_code/01_simulation_app/fire_cli.py` is the main entry point for simulation, reporting, and the lightweight export initializer.
* `1_project_code/02_dataset_generation/gee_collect_preprocess.py` is the fuller Earth Engine collection and preprocessing entry point copied from `project_v2`.
* `1_project_code/02_dataset_generation/aoi_inputs` contains the bundled AOI GeoJSON files used by the included case definitions.
* `2_project_outputs` contains only the final article-backed outputs retained for this package.
* `2_project_outputs/reports_final/reports_5_drones_corridor_scan/figures` contains the retained figure sources used to support the manuscript's final Figures 3 and 4.
* `2_project_outputs/reports_final/reports_5_drones_corridor_scan/tables` contains the retained daily and summary CSV tables used to support the manuscript's reported quantitative results.
* `3_project_article` contains the submission-ready LaTeX manuscript, bibliography, and media assets.

## Code Information

The main simulation entry point is `1_project_code/01_simulation_app/fire_cli.py`.

Supported commands include:

* `inspect` for checking the prepared legacy project layout and validation findings.
* `ca` for running the standalone cellular automaton baseline.
* `hybrid` for running the CA+MAS experiment and returning JSON metrics.
* `report` for producing figure and table outputs from a hybrid run.
* `export-gee` for initializing a Google Earth Engine export workflow from YAML case definitions.

The numbered dataset-generation stage is included because the reworked manuscript project contains only a partial export initializer. The fuller upstream export logic for Earth Engine collection and preprocessing was therefore copied into this package from `project_v2` so that the released code bundle is more complete for sharing and repository publication.

## Usage Instructions

The simulation code expects a prepared raster dataset with static layers and daily time-series rasters arranged in the legacy project layout.

Typical usage pattern:

* Inspect an existing prepared project layout.
* Run the CA baseline over a chosen forecast window.
* Run the hybrid CA+MAS configuration with the verified or modified runtime settings.
* Generate report figures and CSV summaries.
* Use `02_dataset_generation/gee_collect_preprocess.py` if you need the fuller Earth Engine collection and preprocessing workflow.

Example commands:

```bash
python 1_project_code/01_simulation_app/fire_cli.py inspect --project-root path/to/legacy_project
python 1_project_code/01_simulation_app/fire_cli.py ca --project-root path/to/legacy_project --start-date 20210814 --end-date 20210818 --use-barriers
python 1_project_code/01_simulation_app/fire_cli.py hybrid --project-root path/to/legacy_project --start-date 20210814 --end-date 20210818 --use-barriers --drone-count 5 --vision 8 --move 35 --steps-per-day 24
python 1_project_code/01_simulation_app/fire_cli.py report --project-root path/to/legacy_project --start-date 20210814 --end-date 20210818 --use-barriers --drone-count 5 --vision 8 --move 35 --steps-per-day 24 --output-dir ../2_project_outputs/reports_final/reports_5_drones_corridor_scan
python 1_project_code/02_dataset_generation/gee_collect_preprocess.py --config 1_project_code/02_dataset_generation/cases_config.json --case dixie_fire_2021 --dry-run
```

## Requirements

Core Python dependencies used by the packaged code include:

* Python 3.10 or newer is recommended.
* numpy
* rasterio
* scipy
* PyYAML
* earthengine-api
* geemap

Depending on the environment and the parts of the workflow you run, you may also need a working GDAL/rasterio stack and authenticated Google Earth Engine access.

## Package Scope

This repository package was intentionally cleaned before release.

* Only the final article-backed figures and tables were retained in `2_project_outputs`.
* Intermediate ablation, archive, debug, and unused final-output variants were removed.
* The article folder contains the submission-ready LaTeX source and media rather than the full working manuscript history.

## Keywords

Wildfire front forecasting, cellular automaton, multi-agent system, UAV observation, data assimilation, Dixie Fire, geospatial simulation.
