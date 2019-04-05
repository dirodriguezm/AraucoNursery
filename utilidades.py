import os
import rasterio
from osgeo import gdal, ogr, osr
# import cv2
import numpy as np
import pandas as pd


def get_gaussian_kernel(fs_x, fs_y, sigma):
    """
    Tomado de: https://github.com/aditya-vora/crowd_counting_tensorflow
    Create a 2D gaussian kernel
    :param fs_x: filter width along x axis
    :param fs_y: filter width along y axis
    :param sigma: gaussian width
    :return: 2D Gaussian filter of [fs_y x fs_x] dimension
    """

    gaussian_kernel_x = cv2.getGaussianKernel(ksize=np.int(fs_x), sigma=sigma)
    gaussian_kernel_y = cv2.getGaussianKernel(ksize=np.int(fs_y), sigma=sigma)
    gaussian_kernel = gaussian_kernel_y * gaussian_kernel_x.T
    return gaussian_kernel


def get_density_map_gaussian(points, d_map_h, d_map_w, sigma=4):
    """
    Tomado de: https://github.com/aditya-vora/crowd_counting_tensorflow
    Creates density maps from ground truth point locations
    :param points: [x,y] x: along width, y: along height
    :param d_map_h: height of the density map
    :param d_map_w: width of the density map
    :return: density map

    get_density_map_gaussian(coordenadas,raster.height,raster.width,sigma = 3)
    """

    im_density = np.zeros(shape=(d_map_h, d_map_w), dtype=np.float32)

    if np.shape(points)[0] == 0:
        sys.exit()
    for i in range(np.shape(points)[0]):

        f_sz = 15
#         sigma = 4

        gaussian_kernel = get_gaussian_kernel(f_sz, f_sz, sigma)

        x = min(d_map_w, max(1, np.abs(np.int32(np.floor(points[i][0])))))
        y = min(d_map_h, max(1, np.abs(np.int32(np.floor(points[i][1])))))

        if(x > d_map_w or y > d_map_h):
            continue

        x1 = x - np.int32(np.floor(f_sz / 2))
        y1 = y - np.int32(np.floor(f_sz / 2))
        x2 = x + np.int32(np.floor(f_sz / 2))
        y2 = y + np.int32(np.floor(f_sz / 2))

        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0

        change_H = False

        if(x1 < 1):
            dfx1 = np.abs(x1)+1
            x1 = 1
            change_H = True

        if(y1 < 1):
            dfy1 = np.abs(y1)+1
            y1 = 1
            change_H = True

        if(x2 > d_map_w):
            dfx2 = x2 - d_map_w
            x2 = d_map_w
            change_H = True

        if(y2 > d_map_h):
            dfy2 = y2 - d_map_h
            y2 = d_map_h
            change_H = True

        x1h = 1+dfx1
        y1h = 1+dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2

        if (change_H == True):
            f_sz_y = np.double(y2h - y1h + 1)
            f_sz_x = np.double(x2h - x1h + 1)

            gaussian_kernel = get_gaussian_kernel(f_sz_x, f_sz_y, sigma)

        im_density[y1-1:y2, x1-1:x2] = im_density[y1 -
                                                  1:y2, x1-1:x2] + gaussian_kernel
    return im_density


def get_coordenadas(raster, puntos, url_salida, nombre):
    """
    Dado un raster y un shapefile obtiene las posiciones de los puntos con respecto a x e y de la imagen.
    Genera un csv de la forma x, y (lat, lon)
    :raster: corresponde a una imagen tiff abierta con rasterio
    :puntos: Corresponde a un shapefile abierto en geopandas
    :url_salida: carpeta donde se guardara el csv 
    :nombre: nombre del csv con los puntos
    :return: un array con los puntos de la forma y,x
    """
    coordenadas = []
    pixeles = pd.DataFrame(columns=['x', 'y'])
    for index in range(puntos.shape[0]):
        pl = raster.index(puntos["geometry"][index].bounds[0],
                          puntos["geometry"][index].bounds[1])
        pixeles.loc[index] = [pl[0], pl[1]]
    pixeles.to_csv(url_salida+"/"+nombre+".csv")

    for element in range(pixeles.shape[0]):
        coordenadas.append([pixeles["y"][element], pixeles["x"][element]])

    return coordenadas


def array_to_shp(matriz, raster, outputh_file, raster_NoDataValue=0):
    '''
    Adaptado desde: https://github.com/kidpixo/python_public_repository/blob/master/array_to_shapefile.py
    :matriz: numpy array con los datos, debe contener valores enteros, no pueden ser valores flotantes
    :raster: archivo tiff de la region a procesar
    :outputh_file: nombre del archivo que se generara debe contener la extension .shp en el nombre
    :raster_NoDataValue: valor definido como nodata por defecto 0
    :return: nada
    '''

    # obtenemos informacion del raster, para poder calcular la geotransformacion
    nrows, ncols = raster.shape
    xmin, ymin, xmax, ymax = raster.bounds
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransformation = (xmin, xres, 0, ymax, 0, -yres)

    # obtenemos la projeccion en la que se encuentra el raster
    meta = raster.profile
    projection = int(meta['crs']['init'][5:])

    # esto no deberia ser necesario, ya que la matriz recibe los datos formateados
    # plot_matrix[np.isfinite(plot_matrix) == 0] = NoDataValue
    # plot_matrix = plot_matrix.astype('int8')

    # obtenemos el numero de clases presente en la matriz, sin contar el 0, porque esta definido como nodata
    classes = np.unique(matriz)[1:]

   ######################## PASO 1 ######################################
   # Creamos el raster de 1 banda en memoria
    driver = gdal.GetDriverByName('MEM')

    # GDT_Byte : Eight bit unsigned integer
    gdal_datasource = driver.Create(
        '', matriz.shape[1], matriz.shape[0], 1, gdal.GDT_Byte)
    gdal_datasource.SetGeoTransform(geotransformation)

   # set Spatial Reference System
    srs = osr.SpatialReference()  # import srs
    srs.ImportFromEPSG(projection)

    gdal_datasource.SetProjection(srs.ExportToWkt())  # set the data source srs

    # get the 1st raster band, starting from 1, see http://www.gdal.org/classGDALDataset.html#ad96adcf07f2979ad176e37a7f8638fb6
    raster_band = gdal_datasource.GetRasterBand(1)
    raster_band.SetNoDataValue(raster_NoDataValue)  # set the NoDataValues
    raster_band.WriteArray(matriz)

    ######################### PASO 2 #########################################
    # Creamos una capa vectorial en memoria
    drv = ogr.GetDriverByName('Memory')
    ogr_datasource = drv.CreateDataSource('out')

    # create a new layer to accept the ogr.wkbPolygon from gdal.Polygonize
    input_layer = ogr_datasource.CreateLayer(
        'polygonized', srs, ogr.wkbPolygon)

    # add a field to put the classes in
    # see OGRFieldType > http://www.gdal.org/ogr/ogr__core_8h.html#a787194bea637faf12d61643124a7c9fc
    # OFTInteger : Simple 32bit integer
    field_defn = ogr.FieldDefn('class', ogr.OFTInteger)
    input_layer.CreateField(field_defn)  # add the field to the layer

    # create "vector polygons for all connected regions of pixels in the raster sharing a common pixel value"
    # see documentation : www.gdal.org/gdal_polygonize.html
    gdal.Polygonize(raster_band, raster_band.GetMaskBand(), input_layer,  0)

    #############################PASO 3#######################################
    # create 1 bands in raster file

    layerDefinition = input_layer.GetLayerDefn()

    driver = ogr.GetDriverByName("ESRI Shapefile")

    # select the field name to use for merge the polygon from the first and unique field in input_layer
    field_name = layerDefinition.GetFieldDefn(0).GetName()

    # Remove output shapefile if it already exists
    if os.path.exists(outputh_file):
        driver.DeleteDataSource(outputh_file)
    out_datasource = driver.CreateDataSource(outputh_file)
    # create a new layer with wkbMultiPolygon, Spatial Reference as middle OGR file = input_layer
    multi_layer = out_datasource.CreateLayer(
        "merged", input_layer.GetSpatialRef(), ogr.wkbMultiPolygon)

    # Add the fields we're interested in
    # add a Field named field_name = class
    field_field_name = ogr.FieldDefn(field_name, ogr.OFTInteger)
    multi_layer.CreateField(field_field_name)

    # #print out the field name defined
    multylayerDefinition = multi_layer.GetLayerDefn()

    for i in classes:
        # select the features in the middle OGR file with field_name == i
        input_layer.SetAttributeFilter("%s = %s" % (field_name, i))
        multi_feature = ogr.Feature(
            multi_layer.GetLayerDefn())  # generate a feature
        # generate a polygon based on layer unique Geometry definition
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for feature in input_layer:
            # aggregate all the input geometry sharing the class value i
            multipolygon.AddGeometry(feature.geometry())
        # add the merged geoemtry to the current feature
        multi_feature.SetGeometry(multipolygon)
        # add the current feature to the layer
        multi_layer.CreateFeature(multi_feature)
        multi_feature.Destroy()  # desrtroy the current feature

    gdal_datasource = None
    ogr_datasource = None
    out_datasource = None
