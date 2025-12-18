from dtcc_lod2_roofer import roofer

from dtcc_core.model import Building, Mesh, Surface, GeometryType, City, PointCloud
from dtcc_core.common.dtcc_logging import init_logging

from dtcc_core.builder import extrude_building
import numpy as np


debug, info, warning, error, critical = init_logging("dtcc-3dbag-roofer")


def _coords_to_ring(coords, x_offset=0, y_offset=0):
    return [
        [float(float(coord[0]) + x_offset), float(float(coord[1]) + y_offset), 0.0]
        for coord in coords
    ]


def _polygon_to_rings(geom, x_offset=0, y_offset=0):
    # Extract exterior ring
    exterior_ring = _coords_to_ring(geom.exterior.coords, x_offset, y_offset)

    # Extract interior rings (if any)
    interior_rings = []
    # interior_rings = [
    #     _coords_to_ring(interior.coords, x_offset, y_offset)
    #     for interior in geom.interiors
    # ]

    # Combine exterior and interior rings into a single list
    all_rings = [exterior_ring] + interior_rings

    return all_rings


def build_lod2_building(
    building: Building,
    complexity: float = 0.7,
    plane_detect_k: int = 15,
    default_ground_height=0,
) -> bool:

    roof_points = building.point_cloud
    if roof_points is None or len(roof_points) < 1:
        warning("No roof points found in building. Have you extracted the roof points?")
        return False

    building_geometry = building.lod0
    if building_geometry is None:
        building_geometry = building.lod1
    if building_geometry is None:
        warning(
            "No building LOD0 or LOD1 building geometry found. Have you extracted the building geometry?"
        )
        return False
    footprint = building_geometry.to_polygon()
    if (
        footprint is None
        or footprint.is_empty
        or footprint.area == 0
        or footprint.geom_type != "Polygon"
    ):
        warning("Could not create footprint, make sure geometry is valid")
        return False

    ground_height = building.attributes.get("ground_height", default_ground_height)
    roofer_geom, reconstruct_lod = building_roofer(
        roof_points,
        footprint,
        ground_height=ground_height,
        complexity=complexity,
        plane_detect_k=plane_detect_k,
    )

    if roofer_geom is None:
        warning("Roofer failed to create geometry")
        return False

    building.add_geometry(roofer_geom, reconstruct_lod)
    return True


def building_roofer(
    roof_points: PointCloud,
    footprint,
    ground_height: float = 0,
    complexity: float = 0.7,
    plane_detect_k: int = 15,
    return_mesh=False,
) -> (Building, GeometryType):

    if roof_points is None or len(roof_points) < plane_detect_k // 2:
        warning("insufficient roofpoints found in building, skipping roofer")
        return None, None

    if footprint is None:
        warning("no footprint found in building, skipping roofer")
        return None, None
    if footprint.geom_type != "Polygon":
        warning("footprint is not a polygon, skipping roofer")
        return None, None
    roofer_config = roofer.ReconstructionConfig()
    roofer_config.complexity_factor = complexity
    roofer_config.plane_detect_k = plane_detect_k
    roofer_config.plane_detect_min_points = plane_detect_k

    pts = roof_points.points
    x_offset = -np.min(pts[:, 0])
    y_offset = -np.min(pts[:, 1])
    z_offset = -ground_height

    offset = np.array([x_offset, y_offset, z_offset])

    pts += offset
    roofer_config.floor_elevation = 0
    roofer_config.override_with_floor_elevation = True
    # print(f"roof_points: {roof_points[:10]}")
    footprint_rings = _polygon_to_rings(footprint, x_offset, y_offset)
    # print(f"footprint_rings: {footprint_rings}")
    try:
        roofer_meshes = roofer.reconstruct(pts, [], footprint_rings, roofer_config)
        reconstruct_LOD = GeometryType.LOD2
    except RuntimeError as e:
        warning(f"Failed to reconstruct building {e}: trying relaxed conditions")
        roofer_config.complexity_factor = 0.5
        roofer_config.plane_detect_k = max(plane_detect_k // 2, 3)
        roofer_config.plane_detect_min_points = roofer_config.plane_detect_k
        try:
            roofer_meshes = roofer.reconstruct(pts, [], footprint_rings, roofer_config)
            reconstruct_LOD = GeometryType.LOD2
        except:
            warning(f"Failed to reconstruct building {e}: skipping building")
            return None, None
    vertices, faces = roofer.triangulate_mesh(roofer_meshes[0])

    mesh = Mesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.offset(-offset)
    pts -= offset
    if return_mesh:
        return mesh, reconstruct_LOD
    else:
        return mesh.to_multisurface(), reconstruct_LOD
