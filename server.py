from flask import Flask, render_template, flash, redirect, session, url_for, request, g, Markup, jsonify
import requests
import imageio
import torch
import time
import matplotlib.pyplot as plt

from core import Product

app = Flask(__name__)

name = "GHRSST_L4_MUR_Sea_Surface_Temperature"
product = Product(name, offset = 25.0, scale = 0.001, device="cuda") # 298.15

@app.route("/wmts")
def wmts():
    args = request.args
    tilecol = int(args["TileCol"])
    tilerow = int(args["TileRow"])
    tilematrix = int(args['TileMatrix'])

    with torch.no_grad():
        # s = time.time()
        tile = product.gettile(tilecol, tilerow, tilematrix, config = {key : args[key] for key in ["device", "min_value", "max_value", "use_cache"] if key in args})
        # e = time.time()

        # print("Retrieving tilecol {} tilerow {} tilematrix {} took {} seconds".format(tilecol, tilerow, tilematrix, e - s))

        if tile is None:
            flask.abort(404)
        
        tile = imageio.imwrite(imageio.RETURN_BYTES, tile, format="png")

    return tile, 200, {'Content-Type' : 'image/png'}
    
if __name__ == "__main__":
    while True:
        print("enter tilecolumn: ", end="")
        tilecol = int(input())

        print("enter tilerow: ", end="")
        tilerow = int(input())

        print("enter tilematrix: ", end="")
        tilematrix = int(input())

        s = time.time()
        tile = product.gettile(tilecol, tilerow, tilematrix, {"method" : "avg", "device" : "cuda"}) # 
        e = time.time()

        print("Retrieving tilecol {} tilerow {} tilematrix {} took {} seconds".format(tilecol, tilerow, tilematrix, e - s))

        # imageio.imwrite("exampledown.png", tile)
        if tile is not None:
            plt.imshow(tile)
            plt.show()

        del tile