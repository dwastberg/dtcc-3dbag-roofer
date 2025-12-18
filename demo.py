from dtcc_core import io, builder, model
from dtcc_core.model import PointCloud

from dtcc_3dbag_roofer import build_lod2_building
from dtcc_core.builder import build_lod1_buildings

from pathlib import Path
import numpy as np
import sys
import time
from tqdm import tqdm

import dtcc_viewer

# data_dir = Path("/Users/dwastberg/repos/dtcc2/dtcc/data/helsingborg-harbour-2022/")
data_dir = Path("/Users/dwastberg/repos/dtcc2/dtcc/data/helsingborg-residential-2022/")

las_file = Path(
     data_dir / "pointcloud.las"
)
footprint_file = Path(
    data_dir / "footprints.shp"
)

start_time = time.time()
pc = io.load_pointcloud(las_file)
bad_poinst = np.logical_or(pc.points[:, 2] < 0, pc.points[:, 2] > 130)
pc = pc.remove_points(bad_poinst)
pc = pc.remove_global_outliers(5.0)

footprints = io.load_footprints(footprint_file)

# footprints = builder.merge_building_footprints(
#     footprints, lod=model.GeometryType.LOD0, max_distance=0.1, min_area=10
# )

buildings = builder.extract_roof_points(
    footprints, pc, statistical_outlier_remover=True
)
terrain_raster = builder.build_terrain_raster(
    pc, cell_size=2, radius=3, ground_only=True
)

terrain_mesh = builder.build_terrain_mesh(terrain_raster)

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

# print(footprint.wkt)

pc = b.point_cloud

roof_points = PointCloud()



# pc.save("sandbox/testcase/harbour_building_2.las")

# for b in buildings:
#     rpc= b.point_cloud
#     if rpc is not None and len(rpc) > 0:
#         roof_points.merge(rpc)
#         if roof_points.bounds.xmin == 0:
#             print(roof_points.bounds)
#             print(rpc.bounds)
#             print(roof_points.points)
#             sys.exit(1)
# roof_points.view()
# sys.exit(0)

# print(f"Building has {len(pc)} points")
# b, reconstruct_lod = building_roofer(b, complexity=0.8, plane_detect_k=12)
# b.view()

for b in buildings:
    cstart = time.time()
    succ = build_lod2_building(b, complexity=0.8, plane_detect_k=12)
    if not succ:
        fail_cnt += 1
        build_lod1_buildings([b])
        continue

    constuct_time.append(time.time() - cstart)

print(f"Failure count: {fail_cnt}/ {len(buildings)}")

print("Average construction time: ", np.mean(constuct_time))
print("Median construction time: ", np.median(constuct_time))
print("standard deviation construction time: ", np.std(constuct_time))
print("Max construction time: ", np.max(constuct_time))
print("Min construction time: ", np.min(constuct_time))
print("Total construction time: ", np.sum(constuct_time))

print("Number of buildings: ", len(buildings))

city = model.City()
city.add_buildings(buildings)
city.add_terrain(terrain_mesh)
# city.add_geometry(roof_points, model.GeometryType.POINT_CLOUD)
city.view()
