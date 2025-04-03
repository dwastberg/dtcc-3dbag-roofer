import pytest

from dtcc_core import io, builder, model
from dtcc_3dbag_roofer import building_roofer
from pathlib import Path
import numpy as np

las_file = Path(
    "/Users/dwastberg/repos/dtcc2/dtcc/data/helsingborg-residential-2022/pointcloud.las"
)
footprint_file = Path(
    "/Users/dwastberg/repos/dtcc2/dtcc/data/helsingborg-residential-2022/footprints.shp"
)

pc = io.load_pointcloud(las_file)
bad_poinst = np.logical_or(pc.points[:, 2] < 0, pc.points[:, 2] > 130)
pc = pc.remove_points(bad_poinst)
pc = pc.remove_global_outliers(5.0)
footprints = io.load_footprints(footprint_file)

merged_footprints = builder.merge_building_footprints(
    footprints, lod=model.GeometryType.LOD0, max_distance=0.1, min_area=10
)

buildings = builder.extract_roof_points(
    merged_footprints, pc, statistical_outlier_remover=True
)
terrain_raster = builder.build_terrain_raster(
    pc, cell_size=2, radius=3, ground_only=True
)
builder.compute_building_heights(buildings, terrain_raster)


def test_good_building():
    b = buildings[0]
    b, reconstruct_lod = building_roofer(b, plane_detect_k=12)
    assert reconstruct_lod == model.GeometryType.LOD2
