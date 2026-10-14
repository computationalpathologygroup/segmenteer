import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon
from segmenteer.core.utils import mask_to_geojson

def _illustrate_situation():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Create a 200x200 binary mask
    mask = np.zeros((200, 200), dtype=np.uint8)

    # Draw a large outer circle for the donut
    y, x = np.ogrid[:200, :200]
    outer_circle = (x - 100)**2 + (y - 100)**2 <= 80**2

    # Draw three inner circles (holes) inside the donut
    hole1 = (x - 100)**2 + (y - 100)**2 <= 10**2
    hole2 = (x - 130)**2 + (y - 100)**2 <= 10**2
    hole3 = (x - 70)**2 + (y - 100)**2 <= 10**2

    # Create the donut: outer circle minus all holes
    donut = outer_circle & ~hole1 & ~hole2 & ~hole3
    mask[donut] = 1  # Donut = 1

    # Place a small dot inside only the first hole (hole1)
    dot = (x - 100)**2 + (y - 100)**2 <= 5**2
    mask[dot] = 2  # Dot = 2

    # Define a colormap: 0=white, 1=blue (donut), 2=red (dot)
    cmap = ListedColormap(['white', 'blue', 'red'])

    # Plot the mask
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap=cmap, interpolation='nearest')
    plt.title("Donut with 3 Holes and a Dot in One Hole")
    plt.axis('off')

    # Save the plot as a PNG
    plt.savefig('donut_with_holes_and_dot.png', dpi=100, bbox_inches='tight')
    plt.close()


def test_mask_to_geojson_donut_and_dot():
    # Create a 100x100 binary mask
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Draw a donut (outer circle minus inner circle)
    y, x = np.ogrid[:100, :100]
    outer_circle = (x - 50)**2 + (y - 50)**2 <= 30**2
    inner_circle = (x - 50)**2 + (y - 50)**2 <= 10**2
    donut = outer_circle & ~inner_circle
    mask[donut] = 1

    # Draw a small dot inside the donut hole
    dot = (x - 50)**2 + (y - 50)**2 <= 3**2
    mask[dot] = 2  # Use a different value to separate the dot from the donut

    # Convert the mask to GeoJSON
    geojson = mask_to_geojson(mask)

    # Load the GeoJSON into a GeoDataFrame for analysis
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])

    # Check the number of features
    assert len(gdf) == 2, f"Expected 2 features, got {len(gdf)}"

    # Check the donut polygon has an outer and inner ring
    donut_poly = [f for f in gdf.geometry if f.is_valid and f.geom_type == 'Polygon' and len(f.interiors) > 0]
    assert len(donut_poly) == 1, f"Expected 1 donut polygon with hole, got {len(donut_poly)}"

    # Check the dot polygon has only an outer ring
    dot_poly = [f for f in gdf.geometry if f.is_valid and f.geom_type == 'Polygon' and len(f.interiors) == 0]
    assert len(dot_poly) == 1, f"Expected 1 dot polygon without hole, got {len(dot_poly)}"

    # Check for overlapping polygons
    for i, poly1 in enumerate(gdf.geometry):
        for j, poly2 in enumerate(gdf.geometry):
            if i != j and poly1.overlaps(poly2):
                raise AssertionError(f"Polygons {i} and {j} overlap")

    print("All tests passed!")

def test_mask_to_geojson_multi_hole_donut_and_dot():
    # Create a 200x200 binary mask
    mask = np.zeros((200, 200), dtype=np.uint8)

    # Draw a large outer circle for the donut
    y, x = np.ogrid[:200, :200]
    outer_circle = (x - 100)**2 + (y - 100)**2 <= 80**2

    # Draw multiple inner circles (holes) inside the donut
    hole1 = (x - 100)**2 + (y - 100)**2 <= 10**2
    hole2 = (x - 130)**2 + (y - 100)**2 <= 10**2
    hole3 = (x - 70)**2 + (y - 100)**2 <= 10**2

    # Create the donut: outer circle minus all holes
    donut = outer_circle & ~hole1 & ~hole2 & ~hole3
    mask[donut] = 1

    # Place a small dot inside only one of the holes (hole1)
    dot = (x - 100)**2 + (y - 100)**2 <= 5**2
    mask[dot] = 2  # Use a different value for the dot

    # Convert the mask to GeoJSON
    geojson = mask_to_geojson(mask)

    # Load the GeoJSON into a GeoDataFrame for analysis
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])

    # Check the number of features
    assert len(gdf) == 2, f"Expected 2 features (donut + dot), got {len(gdf)}"

    # Identify the donut and dot polygons
    donut_poly = None
    dot_poly = None

    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            if len(geom.interiors) > 0:  # Donut has holes
                donut_poly = geom
            else:  # Dot has no holes
                dot_poly = geom
        elif geom.geom_type == 'MultiPolygon':
            # If the function returns a MultiPolygon for the donut, handle it
            for poly in geom.geoms:
                if len(poly.interiors) > 0:
                    donut_poly = poly
                else:
                    dot_poly = poly

    # Check the donut has multiple holes
    assert donut_poly is not None, "Donut polygon not found"
    assert len(donut_poly.interiors) == 3, f"Expected donut with 3 holes, got {len(donut_poly.interiors)}"

    # Check the dot has no holes
    assert dot_poly is not None, "Dot polygon not found"
    assert len(dot_poly.interiors) == 0, f"Expected dot with 0 holes, got {len(dot_poly.interiors)}"

    # Check the dot is inside one of the donut's holes
    found_dot_in_one_hole = False
    for interior in donut_poly.interiors:
        if Polygon(interior.coords).contains(dot_poly) and not found_dot_in_one_hole:
            found_dot_in_one_hole = True
        elif Polygon(interior.coords).contains(dot_poly) and found_dot_in_one_hole:
            found_dot_in_one_hole = False  # we found it in more holes than 1 which shouldn't be possible.
            break

    assert found_dot_in_one_hole, "Dot is not inside a donut hole"

    # Check for overlapping polygons
    for i, poly1 in enumerate(gdf.geometry):
        for j, poly2 in enumerate(gdf.geometry):
            if i != j and poly1.overlaps(poly2):
                raise AssertionError(f"Polygons {i} and {j} overlap")

    print("All tests passed!")
