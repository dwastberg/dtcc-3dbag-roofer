from dtcc_core import io, builder, model
from dtcc_3dbag_roofer import building_roofer
from pathlib import Path
import numpy as np

import time
from tqdm import tqdm

import dtcc_viewer

las_file = Path(
    "/Users/dwastberg/repos/dtcc2/dtcc/data/helsingborg-harbour-2022/pointcloud.las"
)
footprint_file = Path(
    "/Users/dwastberg/repos/dtcc2/dtcc/data/helsingborg-harbour-2022/footprints.shp"
)

start_time = time.time()
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

lod2_cnt = 0
lod1_cnt = 0
fail_cnt = 0
new_buildings = []
constuct_time = []
bad_residential_points = (32, 64, 118, 124)
# bad_harbour_points = (2, 4, 7)

b = buildings[2]

footprint = b.get_footprint().to_polygon()

print(f"is valid: {footprint.is_valid}")

print(footprint.wkt)

pc = b.point_cloud

pc.save("sandbox/testcase/harbour_building_2.las")

print(f"Building has {len(pc)} points")
# b, reconstruct_lod = building_roofer(b, complexity=0.8, plane_detect_k=12)
# b.view()

# for idx, b in enumerate(buildings):
#     print(idx)
#     # if idx in bad_harbour_points:
#     #     # these buildings cause roofer to hang for some reason
#     #     continue
#     cstart_time = time.time()
#     footprint = b.get_footprint().to_polygon()
#     if (
#         (not footprint.is_valid)
#         or (footprint.area < 10)
#         or footprint.geom_type != "Polygon"
#     ):
#         print(f"Building {idx} has an invalid footprint")
#         print(f"Area: {footprint.area}")
#         print(f"Geom type: {footprint.geom_type}")

#     b, reconstruct_lod = building_roofer(b, plane_detect_k=12)
#     if reconstruct_lod is None:
#         fail_cnt += 1
#         continue
#     elif reconstruct_lod == model.GeometryType.LOD2:
#         lod2_cnt += 1
#     elif reconstruct_lod == model.GeometryType.LOD1:
#         lod1_cnt += 1
#     if reconstruct_lod is None:
#         ms = builder.extrude_building(b, always_use_default=True)
#         # b.add_geometry(ms, model.GeometryType.LOD1)
#     new_buildings.append(b)
#     constuct_time.append(time.time() - cstart_time)

# print("--- %s seconds ---" % (time.time() - start_time))
# print("LOD2: ", lod2_cnt)
# print("LOD1: ", lod1_cnt)
# print("Failed: ", fail_cnt)

# print("Average construction time: ", np.mean(constuct_time))
# print("Median construction time: ", np.median(constuct_time))
# print("standard deviation construction time: ", np.std(constuct_time))
# print("Max construction time: ", np.max(constuct_time))
# print("Min construction time: ", np.min(constuct_time))

# # city = model.City()
# # city.add_buildings(new_buildings)
# # city.view()
