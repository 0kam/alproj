import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import numpy as np
import os
import datatable as dt
import pandas as pd
import sqlite3
import math
from alproj.optimize import extrinsic_mat, distort


def create_db(aerial, dsm, out_path, res=1.0, chunksize=10000):
    """
    Creates a SQLite3 database of a colored surface from a Digital Surface Model and an ortho-rectificated aerial/satellite photograph.
    The result database contains coordinates, colors, and index data of created surface. The given DSM and aerial/satellite photograph must be 
    transformed to the same planar coordinate reference system (such as UTM). The unit of these must be m.

    Parameters
    ----------
    aerial : rasterio.DatasetReader
        An aerial photograph opend by rasterio.open()
    dsm : rasterio.DatasetReader
        A Digital SurfaceModel opend by rasterio.open()
    out_path : str
        Path for output SQLite3 file.
    res : float
        Mesh resolution for generated surface in m.
    chunksize : int
        Specify the number of rows in each batch to be written at a time. By default, all rows will be written at once.
        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html

    Returns
    -------

    """
    if os.path.exists(out_path):
        os.remove(out_path)
    t = min([aerial.bounds.top, dsm.bounds.top])
    r = min([aerial.bounds.right, dsm.bounds.right])
    b = max([aerial.bounds.bottom, dsm.bounds.bottom])
    l = max([aerial.bounds.left, dsm.bounds.left])
    aerial2, transform_a = merge([aerial], bounds=[l,b,r,t], res=res, resampling=Resampling.cubic_spline)
    dsm2, transform_d = merge([dsm], bounds=[l,b,r,t], res=res, resampling=Resampling.cubic_spline)
    
    if transform_a == transform_d:
        transform = transform_a
    else:
        print("error in merging aerial photo and DSM")
    
    # xyz
    x = np.arange(0, dsm2.shape[2]) * transform[0] + transform[2]
    y = np.arange(0, dsm2.shape[1]) * transform[4] + transform[5]
    xx, yy = np.meshgrid(x, y)
    w = xx.shape[0]
    h = xx.shape[1]
    zz = np.squeeze(dsm2)
    # RGB
    R = aerial2[0,:,:]
    G = aerial2[1,:,:]
    B = aerial2[2,:,:]
    vertices = np.vstack((xx,yy,zz,R,G,B)).reshape([6, -1])
    vertices = np.vstack((vertices, np.squeeze(np.arange(0, vertices.shape[1], 1, np.int32)))).transpose()
    del(x, y, xx, yy, zz, R, G, B, dsm2, aerial2)
    # save point cloud data as SQLite3 Data Base
    columns = ["x","y","z","r","g","b","id"]
    df = pd.DataFrame(data=vertices, columns=columns, dtype="float64")
    df[["r","g","b"]] = df[["r","g","b"]].astype("uint8")
    df["id"] = df["id"].astype("uint32")
    conn = sqlite3.connect(out_path)
    dtypes = {"id":"Integer", "x":"Float", "y":"Float", "z":"Float","r":"Integer", "g":"Integer", "b":"Integer"}
    df.to_sql('vertices',conn,if_exists='replace',index=None, chunksize=chunksize, dtype = dtypes, method = None)
    del(vertices, df)
    # indices of vertices in each triangle
    ai = np.arange(0, w)
    aj = np.arange(0, h)
    aii, ajj = np.meshgrid(ai, aj)
    a = aii + ajj * h
    a = a.flatten()
    tria = np.vstack((a, a + h, a + h + 1, a, a + h + 1, a + 1))
    tria = np.transpose(tria).reshape([-1, 3])
    del(aj, ai, aii, ajj, a)
    
    # write down into the sqlite
    columns = ["v1", "v2", "v3"]
    df = pd.DataFrame(data=tria, columns=columns, dtype="uint32")
    del(tria)
    df.to_sql('indices',conn,if_exists='replace',index=None, chunksize=chunksize, dtype="Integer")
    del(df)
    conn.close()

def crop(conn, params, distance=3000, chunksize=100000):
    """
    Crops the given surface in fan shape.
    
    Parameters
    ----------
    params : dict
        Camera parameters.
    distance : float default 3000
        Radius of the fan shape in m.
    chunksize : int default 100000
        Specify the number of rows in each batch to be written at a time. By default, all rows will be written at once.
        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    
    Returns
    -------
    vert : numpy.ndarray
        Coordinates of vetices (X, Z, Y).
    col : numpy.ndarray
        Colors of vertices (R, G, B).
    ind : numpy.ndarray
        Index array that shows which three poits shape a triangle. See http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-9-vbo-indexing/ .
    """
    # filter and collect vertices
    corners = np.array([
        [-params["w"]/2, -params["w"]/2, params["w"]/2, params["w"]/2],
        [0, 0, 0, 0],
        [params["h"]/2, -params["h"]/2, params["h"]/2, -params["h"]/2],
        [1, 1, 1, 1]
        ])
    p = params.copy()
    p["x"] = p["y"] = p["z"] = p["pan"] = 0
    emat = extrinsic_mat(p["pan"], p["tilt"], p["roll"], p["x"], p["y"], p["z"])
    corners = np.dot(emat, corners)

    centre = np.array([(p["w"] - 1) / 2, (p["h"] - 1) / 2], dtype = 'float32')
    x1 = corners[0,:] / centre[0]
    y1 = (p["h"] / p["w"]) * corners[1,:] / centre[1]
    corners = np.vstack([x1,y1,np.ones(4)])
    corners = distort(corners, p["a1"], p["a2"], p["k1"], p["k2"], p["k3"], p["k4"], p["k5"], p["k6"], p["p1"], p["p2"], p["s1"], p["s2"], p["s3"], p["s4"])
    x = corners[0, :] * centre[0] ## Modified from coners[0, :] * centre[1]
    fov = params["fov"] /  params["w"] * (max(x) - min(x))
    pan = params["pan"] + math.atan2(( (max(x) + min(x)) / 2 ) / centre[0], 1) * 180 / math.pi

    params = {"x":str(params["x"]),"y":str(params["y"]),"pan":str(pan), "fov":str(fov)}
    csr = conn.cursor()
    conn.create_function("ATAN2", 2, math.atan2)
    conn.create_function("POWER", 2, math.pow)
    csr.execute("SELECT * \
    FROM (SELECT `id`, `x`, `y`, `z`, `r`, `g`, `b` \
    FROM (SELECT * \
    FROM (SELECT `id`, `x`, `y`, `z`, `r`, `g`, `b`, `theta`, CASE WHEN (`theta` < 0.0) THEN (-`theta`) WHEN NOT(`theta` < 0.0) THEN (360.0 - `theta`) END AS `theta2` \
    FROM (SELECT `id`, `x`, `y`, `z`, `r`, `g`, `b`, ATAN2(`y` - " + params["y"] + ", `x` -" + params["x"] + ") * 180.0 / 3.14159265358979 - 90.0 AS `theta` \
    FROM `vertices`)) \
    WHERE ((POWER((`x` - " + params["x"] + "), 2.0) + POWER((`y` - " + params["y"] +"), 2.0) < POWER("+ str(distance) + ", 2.0)) \
        AND (POWER((`x` - " + params["x"] + "), 2.0) + POWER((`y` - " + params["y"] + "), 2.0) > 5.0) AND (`theta2` >= "+ params["pan"] + "-" + params["fov"] + "/ 2.0 * 1.0) \
            AND (`theta2` <= "+ params["pan"] + "+" + params["fov"] + " / 2.0 * 1.0)))) \
    WHERE (((`x`) IS NULL) = 0 AND ((`y`) IS NULL) = 0 AND ((`z`) IS NULL) = 0 AND ((`r`) IS NULL) = 0 AND ((`g`) IS NULL) = 0 AND ((`b`) IS NULL) = 0)") 
    
    vert = dt.Frame(np.array(csr.fetchall()), names = ["id","x","y","z","r","g","b"])
    # collect all indices
    nrow = conn.execute("SELECT count(*) FROM indices").fetchall()[0][0]
    csr = conn.cursor()
    csr.execute("select * from indices")
    ind_full = np.array([])
    for i in range(math.ceil(nrow / chunksize)):
        if i == math.ceil(nrow / chunksize) - 1:
            chunksize = nrow - (chunksize * (i-1))
        x = csr.fetchmany(chunksize)
        ind_full = np.append(ind_full, x)
    ind_full = ind_full.reshape([-1,3]).astype(np.int64)
    ind_full = dt.Frame(ind_full, names = ["v1", "v2", "v3"])
    id_ind = dt.Frame(np.vstack((np.arange(0,vert.nrows,1), vert["id"].to_numpy().squeeze())).astype("int64").transpose(), names = ["ind","id"])
    
    id_ind.names = ["ind", "v1"]
    id_ind.key="v1"
    ind = ind_full[:, :, dt.join(id_ind)]
    
    id_ind.names = ["v2", "ind"]
    id_ind.key = "v2"
    ind = ind[:, :, dt.join(id_ind)]
    
    id_ind.names = ["v3", "ind"]
    id_ind.key="v3"
    ind = ind[:, :, dt.join(id_ind)]
    
    ind = ind[dt.rowall(dt.math.isna(dt.f[:])==False), ["ind", "ind.0", "ind.1"]]
    col = vert[:,["r", "g", "b"]].to_numpy() / 255
    vert = vert[:, ["x", "z", "y"]].to_numpy() # in opengl, z is near-far axis
    ind = ind.to_numpy()
    return vert, col, ind # vertex coordinates, vertex color, index