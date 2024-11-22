from dtcc_core.model import Building, Mesh, Surface
from dtcc_core.common.dtcc_logging import init_logging
import numpy as np

debug, info, warning, error, critical = init_logging("dtcc-3dbag-roofer")

import roofer


def _coords_to_ring(coords, x_offset=0, y_offset=0):
    return [
        [float(coord[0]) + x_offset, float(coord[1]) + y_offset, 0.0]
        for coord in coords
    ]


def _polygon_to_rings(geom, x_offset=0, y_offset=0):
    # Extract exterior ring
    exterior_ring = _coords_to_ring(geom.exterior.coords, x_offset, y_offset)

    # Extract interior rings (if any)
    interior_rings = [
        _coords_to_ring(interior.coords, x_offset, y_offset)
        for interior in geom.interiors
    ]

    # Combine exterior and interior rings into a single list
    all_rings = [exterior_ring] + interior_rings

    return all_rings


def building_roofer(
    building: Building, complexity: float = 0.7, default_ground_height=0
) -> Building:
    roof_points = building.point_cloud
    if roof_points is None or len(roof_points) < 5:
        warning("insufficient roofpoints found in building, skipping roofer")
        return building
    ground_height = building.attributes.get("ground_height", default_ground_height)
    footprint = building.get_footprint()
    if footprint is None:
        warning("no footprint found in building, skipping roofer")
        return building

    roofer_config = roofer.ReconstructionConfig()
    roofer_config.complexity_factor = complexity
    x_offset = -np.min(footprint[:, 0])
    y_offset = -np.min(footprint[:, 1])
    z_offset = -ground_height
    offset = np.array([x_offset, y_offset, z_offset])

    roofer_config.floor_elevation = 0
    roofer_config.override_with_floor_elevation = True

    pts = roof_points.data - offset
