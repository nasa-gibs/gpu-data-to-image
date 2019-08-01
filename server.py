from flask import Flask, render_template, flash, redirect, session, url_for, request, g, Markup, jsonify, abort
import requests
import imageio
import torch
import time
import matplotlib.pyplot as plt

from core import Product

app = Flask(__name__)

name = "AVHRR_OI_L4_GHRSST_NCEI"
product = Product(name, device="cuda") # 298.15

@app.route("/wmts")
def wmts():
    args = request.args
    tilecol = int(args["TileCol"])
    tilerow = int(args["TileRow"])
    tilematrix = int(args['TileMatrix'])

    with torch.no_grad():
        # s = time.time()
        config = {key : args[key] for key in ["device", "min_value", "max_value", "use_cache", 'cmap', 'device', 'filter', 'scale'] if key in args}
        # config["cmap"] = "VIIRS_SNPP_Brightness_Temp_BandI5_Day"
        # config["method"] = "avg"
        # config["filter"] = "sobel"
        tile = product.gettile(tilecol, tilerow, tilematrix, config = config)
        # e = time.time()

        # print("Retrieving tilecol {} tilerow {} tilematrix {} took {} seconds".format(tilecol, tilerow, tilematrix, e - s))

        if tile is None:
            abort(404)
        
        tile = imageio.imwrite(imageio.RETURN_BYTES, tile, format="png")

    return tile, 200, {'Content-Type' : 'image/png'}

@app.route("/getstats")
def stats():
    args = request.args
    bbox = [float(x) for x in args["bbox"].split(",")]

    with torch.no_grad():
        stats = product.getstats(*bbox)

    return jsonify(stats), 200, {'Content-Type' : 'json'}

@app.route("/clearcache")
def clearcache():
    product.cache.clear()
    product.random = None
    return "cache cleared", 200

if __name__ == "__main__":
    while True:
        print("enter tilecolumn: ", end="")
        tilecol = int(input())

        print("enter tilerow: ", end="")
        tilerow = int(input())

        print("enter tilematrix: ", end="")
        tilematrix = int(input())

        s = time.time()
        tile = product.gettile(tilecol, tilerow, tilematrix, {"method" : "avg", "device" : "cuda"})
        e = time.time()

        print("Retrieving tilecol {} tilerow {} tilematrix {} took {} seconds".format(tilecol, tilerow, tilematrix, e - s))

        # imageio.imwrite("exampledown.png", tile)
        if tile is not None:
            plt.imshow(tile)
            plt.show()

        del tile