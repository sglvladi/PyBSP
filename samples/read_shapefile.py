import shapefile
import pickle
from shapely.geometry import shape
import matplotlib.pyplot as plt

shapes = shapefile.Reader("shapefiles/simplified_land_polygons.shp")
#first feature of the shapefile
#feature = shapes.shapeRecords()[0]

features = shapes.shapeRecords()

polygons = []

for i, feature in enumerate(features):
    first = feature.shape.__geo_interface__
    print(i)

    shp_geom = shape(first)
    polygons.append(shp_geom)
    x,y = shp_geom.exterior.xy
    plt.plot(x,y)

# pickle.dump(polygons, open('polygons.p', 'wb'))
plt.show()


