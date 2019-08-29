
import requests
import pickle
import os
import scipy.io.netcdf as netcdf
import numpy as np
import torch

import argparse
import sys

def from_url(url, name, username = None, password = None):
    path = url.split("/")[-1]
    
    if not os.path.exists(path):
        r = requests.get(url, auth=(username, password), stream=True)
        size = int(r.headers["Content-Length"])

        r.raise_for_status()

        dl = 0
        with open(path, 'wb') as handle:
            for block in r.iter_content(4096):
                dl += len(block)
                handle.write(block)

                done = int(50 * dl / size)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

    return path


def decompress(path):
    if path.endswith(".nc"):
        return path
    elif path.endswith(".bz2"):
        import bz2, shutil
        newpath = ".".join(path.split(".")[:-1])
        if not os.path.exists(newpath):
            with bz2.BZ2File(path, "r") as source:
                with open(newpath, "wb") as target:
                    shutil.copyfileobj(source, target)

        return newpath
    else:
        raise ValueError("Compression scheme not supported for downloaded file {}")

def process(path, variable, name, flip=False):
    with netcdf.netcdf_file(os.path.join(path), "r") as file:
        var = file.variables[variable]
        offset = var.add_offset
        scale = var.scale_factor
        missing = var._get_missing_value()
        data = torch.Tensor(var.data.byteswap().view("<i2")).to(torch.int16).squeeze()
        if flip:
            data = data.flip(0)

        raw = data.float() * scale + offset

    if not os.path.exists("data"):
        os.mkdir("data")
    
    picklepath = os.path.join("data", name + ".pickle")
    with open(picklepath, "wb") as f:
        pickle.dump(raw, f)

    print("Pickled file {} with offset {} and scale {}. Range is {} to {}. Shape is {}".format(picklepath, offset, scale, raw.min(), raw.max(), raw.shape))

    return picklepath

def download(url, name, variable, username = None, password = None, flip=False):
    print("downloading from url {}".format(url))
    path = from_url(url, name, username, password)
    print("done downloading. decompressing file.")
    decompressed = decompress(path)
    os.remove(path)
    print("done decompressing. processing")
    filepath = process(decompressed, variable, name, flip=flip)
    print("done processing. file saved to {}".format(filepath))
    os.remove(decompressed)

    return filepath
    
parser = argparse.ArgumentParser(description="download and process netcdf files from NASA data repositories like PO.DAAC")

parser.add_argument('url', type=str, help='url for raw data file (supported formats: netCDF')
parser.add_argument('name', type=str, help='name of product')
parser.add_argument('variable', type=str, help='variable in netcdf file to use')

parser.add_argument('-u', type=str, dest="credentials", default="", help='username and password for website')
parser.add_argument('--filename', type=str, dest="credentials", default="", help='username and password for website')
parser.add_argument('--flip', action='store_true', default=False, help='flip image vertically')

args = parser.parse_args()

if args.credentials == "":
    filepath = download(args.url, args.name, args.variable, flip = args.flip)
else:
    filepath = download(args.url, args.name, args.variable, username = args.credentials.split(":")[0], password = args.credentials.split(":")[1], flip = args.flip)
