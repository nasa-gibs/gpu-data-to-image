import requests
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from core import Product

def get_data(name="AVHRR_OI_L4_GHRSST_NCEI", bbox=(-20, -20, 10, 10), date="2014-01-01"):
    url = "https://oceanworks.jpl.nasa.gov/datainbounds?ds={layer}&startTime={time}T00:00:00Z&endTime={time}T00:00:00Z&b={bbox}"

    xwidth = bbox[2] - bbox[0]
    ywidth = bbox[3] - bbox[1]

    temp_url = url.format(layer=name, time=date, bbox=",".join((str(e) for e in bbox)))
    print(temp_url)

    r = requests.get(temp_url)
    r.raise_for_status()

    json = r.json()
    data = [(entry['latitude'], entry['longitude'], entry['data'][0]['variable']) for entry in json['data']]
    data = np.stack(list(zip(*data)), axis=1)

    xvals = np.round((data[:,0] - data[:,0].min()) / 0.25).astype(np.int64)
    yvals = np.round((data[:,1] - data[:,1].min()) / 0.25).astype(np.int64)
    values = data[:,2]

    print(values.shape)

    tile = np.zeros((xwidth * 4, ywidth * 4))

    tile[xvals, yvals] = values
    tile = tile[::-1,::-1]

    return tile, json

name = "AVHRR_OI_L4_GHRSST_NCEI"
# data, json = get_data(name, bbox=(-65, -65, 65, 65))

# with open(os.path.join("data", name + ".pickle"), "wb") as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

product = Product(name)
product.mrfgen({"output_dir" : "output", "method" : "avg"})