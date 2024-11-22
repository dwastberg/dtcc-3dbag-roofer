from dtcc_core.model import Building, Mesh, Surface, GeometryType
from dtcc_core.common.dtcc_logging import init_logging
import numpy as np

debug, info, warning, error, critical = init_logging("dtcc-3dbag-roofer")

from . import roofer


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
    building: Building,
    complexity: float = 0.7,
    plane_detect_k: int = 15,
    default_ground_height=0,
) -> Building:
    roof_points = building.point_cloud
    if roof_points is None or len(roof_points) < plane_detect_k // 2:
        warning("insufficient roofpoints found in building, skipping roofer")
        return building, None
    ground_height = building.attributes.get("ground_height", default_ground_height)
    footprint = building.get_footprint().to_polygon()
    if footprint is None:
        warning("no footprint found in building, skipping roofer")
        return building, None
    if footprint.geom_type != "Polygon":
        warning("footprint is not a polygon, skipping roofer")
        return building, None
    roofer_config = roofer.ReconstructionConfig()
    roofer_config.complexity_factor = complexity
    roofer_config.plane_detect_k = plane_detect_k
    roofer_config.plane_detect_min_points = plane_detect_k

    roof_points = roof_points.points
    x_offset = -np.min(roof_points[:, 0])
    y_offset = -np.min(roof_points[:, 1])
    z_offset = -ground_height

    offset = np.array([x_offset, y_offset, z_offset])

    roof_points += offset
    roofer_config.floor_elevation = 0
    roofer_config.override_with_floor_elevation = True

    footprint_rings = _polygon_to_rings(footprint, x_offset, y_offset)

    try:
        roofer_meshes = roofer.reconstruct(
            roof_points, [], footprint_rings, roofer_config
        )
        reconstruct_LOD = GeometryType.LOD2
    except RuntimeError as e:
        roofer_config.lod = 12
        roofer_config.complexity_factor = 0.5
        roofer_config.plane_detect_min_points = plane_detect_k // 2
        try:
            roofer_meshes = roofer.reconstruct(
                roof_points, [], footprint_rings, roofer_config
            )
            reconstruct_LOD = GeometryType.LOD1
        except:
            # warning(f"Failed to reconstruct building: skipping building")
            return building, None
        warning("Building reconstructed with LOD1")
    vertices, faces = roofer.triangulate_mesh(roofer_meshes[0])

    mesh = Mesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.offset(-offset)
    building.add_geometry(mesh.to_multisurface(), reconstruct_LOD)
    return building, reconstruct_LOD
