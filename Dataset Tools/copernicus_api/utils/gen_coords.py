import geopandas as gpd
import numpy as np
import polars as pl

from shapely.geometry import Point
from pyproj import Geod
from tqdm import trange

def polygon_area(polygon):
    """Calculate the geodetic area (in m^2) of a polygon using WGS84 ellipsoid."""
    geod = Geod(ellps="WGS84")
    if polygon.is_empty:
        return 0
    if polygon.geom_type == 'Polygon':
        lons, lats = polygon.exterior.coords.xy
        area, _ = geod.polygon_area_perimeter(lons, lats)
        return abs(area)
    elif polygon.geom_type == 'MultiPolygon':
        return sum(polygon_area(poly) for poly in polygon.geoms)
    else:
        return 0

def random_point_in_polygon(polygon):
    """Generate a uniform random point within a given polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        # Random point in bounding box
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        p = Point(lon, lat)
        if polygon.contains(p):
            return p

def sample_uniform_land_point(land_shp):
    # Load landmass polygons
    land = gpd.read_file(land_shp)

    # Convert all geometries to WGS84 (EPSG:4326)
    land = land.to_crs(epsg=4326)

    # Flatten all multipolygons into a list of polygons
    polygons = []
    for geom in land.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == 'Polygon':
            polygons.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            polygons.extend(list(geom.geoms))

    # Calculate geodetic area for each polygon
    areas = np.array([polygon_area(poly) for poly in polygons])
    total_area = np.sum(areas)
    if total_area == 0:
        raise RuntimeError("No land area found!")

    # Choose a polygon weighted by its area
    probs = areas / total_area
    idx = np.random.choice(len(polygons), p=probs)
    chosen_poly = polygons[idx]

    # Generate a random point within the chosen polygon
    pt = random_point_in_polygon(chosen_poly)
    return pt.x, pt.y  # longitude, latitude

if __name__ == "__main__":
    land_shp = "../ne_data/ne_50m_land/ne_50m_land.shp"
    N = 10

    points = [sample_uniform_land_point(land_shp) for _ in trange(N)]
    
    lons, lats = zip(*points)
    df = pl.DataFrame({ "Longitude": lons, "Latitude": lats })
    df.write_csv("../data/Global_spots.csv")