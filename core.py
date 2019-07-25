import scipy.io
import numpy as np
import imageio
import sys
import torch
import time
import pickle

import xmltodict

import sys
import time
import os
import struct
import math
import requests

def getcmap(layer, local=False):
    local_path = os.path.join("colormaps", layer + '.xml')

    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            data = f.read()
    else:
        try:
            base_url = "http://gibs.earthdata.nasa.gov/colormaps/v1.0/{layer}.xml"
            url = base_url.format(layer=layer)
            r = requests.get(url)
            r.raise_for_status()
            data = r.content
        except:
            print("No colormap found for layer {}. Defaulting to default colormap".format(layer))
            with open(os.path.join("colormaps", "default.xml"), "rb") as f:
                data = f.read()

    cmap = xmltodict.parse(data)
    cmap_body = cmap['ColorMap']['ColorMapEntry']

    cmap = np.zeros((214,3), dtype=np.uint8)

    for i, entry in enumerate(cmap_body[1:-1]):
        color = tuple([int(x) for x in entry['@rgb'].split(",")])
        x = [int(x) for x in entry['@rgb'].split(",")]
        cmap[i] = x

    return cmap


def getdata(name): # temporarily, load pickled data. Can also load from NetCDF (same cost).
    with open(os.path.join("data", name + ".pickle"), "rb") as f:
        return torch.Tensor(pickle.load(f))

# offset = file.variables["analysed_sst"].add_offset
# scale = file.variables["analysed_sst"].scale_factor
# missing = file.variables["analysed_sst"]._get_missing_value()

class TileCache:
    def __init__(self, maxsize=1E9, verbose=False):
        self.verbose = verbose
        self._cache = {}
        self._size = 0
        self._maxsize = maxsize

    def clear(self):
        self._cache = {}
        self._size = 0

    def store(self, col, row, matrix, tile):
        if self._size + self._size > self._maxsize:
            self.clear()

        self._size += self.tilesize(tile)
        self._cache["{}-{}-{}".format(col, row, matrix)] = tile

    def get(self, col, row, matrix):
        return self._cache["{}-{}-{}".format(col, row, matrix)]

    def contains(self, col, row, matrix):
        return "{}-{}-{}".format(col, row, matrix) in self._cache

    def tilesize(self, tile):
        return tile.size * tile.itemsize

    def size(self):
        return self._size

class Product:
    def __init__(self, name, data=None, offset=0.0, scale=1.0, device="cuda"):
        self.name = name

        # self.offset = 298.15
        # self.scale = 0.001
        # self.missing = -32768

        self.offset = offset
        self.scale = scale

        self.cmap = torch.Tensor(getcmap(self.name))

        if device == "cuda":
            self.cmap = self.cmap.cuda().to(torch.uint8) # TODO fix this
        else:
            self.cmap = self.cmap.to(torch.uint8) # TODO fix this

        if data is None:
            self.data = torch.Tensor(getdata(self.name))
        else:
            if isinstance(data, torch.Tensor):
                self.data = data
            else:
                self.data = torch.Tensor(data.copy())

        self.cache = TileCache()
        self.shape = self.data.shape

        self.num_overviews = max(1, max(math.floor(math.log2(self.shape[0] / 512)), math.floor(math.log2(self.shape[1] / 512))) + 1)

        print("Loaded Product {} with shape {} and {} overviews".format(self.name, self.shape, self.num_overviews))

    def pickle(self):
        with open(self.name + ".pickle", "wb") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def getshape(self, tilecolumn, tilerow, tilematrix):
        size = 512 * 2 ** (self.num_overviews - tilematrix - 1)
        print(size * tilerow, min(size * (tilerow + 1), self.shape[0]), size * tilecolumn, min(size * (tilecolumn + 1), self.shape[1]))
        return self.data[size * tilerow : min(size * (tilerow + 1), self.shape[0]), size * tilecolumn : min(size * (tilecolumn + 1), self.shape[1])]

    def mrfgen(self, config=None):
        """config:
            device (str) -- cuda or cpu
            min_value (float) -- minimum value to set (C)
            max_value (float) -- maximum value to set (C)
            method (str) -- downsampling method (nn or avg supported)
            format (str) -- image format for mrf (jpeg or png supported)
            output_dir (str) -- output directory for MRF files (current directory is default)
        """

        if config is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            method = "nn"
            format = "jpeg"
            min_value = 0.0
            max_value = 100.0
            output_dir = os.getcwd()
        else:
            if config.get("device", "cpu") == "cuda" and not torch.cuda.is_available():
                raise ValueError("Configuration backend was cuda but cuda is not available")

            device = torch.device("cuda:0") if config.get("device", "cuda") == "cuda" else torch.device("cpu")
            method = config.get("method", "nn")
            format = config.get("format", "jpeg")
            min_value = float(config.get("min_value", 0.0))
            max_value = float(config.get("max_value", 100.0))
            output_dir = config.get("output_dir", os.getcwd())

            if not os.path.exists(output_dir):
                raise FileNotFoundError("output_dir was {} but directory does not exist".format(output_dir))

        print("Running MRF generation on file {} with device {}, method {} and format {}".format(self.name, device, method, format))
        data = self.data.to(device, copy=True).float()
        data = data.mul_(self.scale).add_(self.offset)
        data = data.clamp_(min=min_value, max=max_value)
                    
        if method == "nn":
            data = data.sub_(data.min()).div_(data.max()).mul_(len(self.cmap) - 1).long()
            data = self.cmap[data]

        idx_file = open(os.path.join(output_dir, self.name + ".idx"), "wb")
        
        if format == "jpeg":
            data_file = open(os.path.join(output_dir, self.name + ".pjg"), "wb")
        elif format == "png":
            data_file = open(os.path.join(output_dir, self.name + ".ppg"), "wb")
        elif format == "tiff":
            data_file = open(os.path.join(output_dir, self.name + ".ptf"), "wb")
        else:
            raise ValueError("Format {} is not a supported image format".format(format))

        steps = [(math.ceil(self.shape[0] // 2 ** i / 512), math.ceil(self.shape[1] // 2 ** i / 512)) for i in range(self.num_overviews)]
        print(steps)

        for i, (numrows, numcols) in enumerate(steps):
            if method == "nn":
                scaled = data[:: 2 ** i, :: 2 ** i].cpu().numpy() # can also downsample by a factor of two each time
            elif method == "avg":                
                if i != 0:
                    data = torch.nn.functional.avg_pool2d(data.float().unsqueeze(0), 2).squeeze()
                
                scaled = (data - data.min()).div_(data.max()).mul_(len(self.cmap) - 1).to(torch.int16)
                
                try:
                    scaled = scaled.long()
                    scaled = self.cmap[scaled].cpu().numpy()
                except RuntimeError: # out of GPU memory
                    data = data.cpu()
                    scaled = scaled.long()
                    scaled = self.cmap[scaled].cpu().numpy()
                    data = data.to(device)

            else:
                raise ValueError("{} is not a supported downsampling method".format(method))

            ax, ay = np.meshgrid(np.arange(numrows), np.arange(numcols))
            entries = np.stack([ax.flatten(), ay.flatten()], axis=1)
            print(i, numrows, numcols, 2 ** i)

            for j, (row, col) in enumerate(entries):
                tile = scaled[row * 512 : (row + 1) * 512, col * 512 : (col + 1) * 512]

                empty = np.zeros((512, 512, 3), dtype=np.uint8)
                empty[0 : tile.shape[0], 0: tile.shape[1]] = tile
                # breakpoint()  
                
                pos = data_file.tell()
                imageio.imwrite(data_file, empty, format=format) # TODO speed this up with multiprocessing
                # success, buffer = cv2.imencode(".jpg", empty)
                # buffer.tofile(data_file)
                size = data_file.tell() - pos
                idx_file.write(struct.pack(">Q", pos))
                idx_file.write(struct.pack(">Q", size))
                
        idx_file.close()
        data_file.close()

        with open(os.path.join("templates", "template.mrf"), "r") as f:
            mrf = f.read()
            mrf = mrf.format(sizex=self.shape[1], sizey=self.shape[0], format=format.upper())
        
        with open(os.path.join(output_dir, self.name + ".mrf"), "w") as f:
            f.write(mrf)
        
    def gettile(self, tilecolumn, tilerow, tilematrix, config=None):
        """config:
            device (str) -- cuda or cpu
            min_value (float) -- minimum value to set (C)
            max_value (float) -- maximum value to set (C)
            use_cache (bool) -- should use cache regardless of configuration
            scale (bool) -- should scale data using min_value and max_value
        """
        
        if config is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            min_value = 0.0
            max_value = 100.0
            use_cache = True
            method = "nn"
            scale = False
        else:
            if config.get("device", "cpu") == "cuda" and not torch.cuda.is_available():
                raise ValueError("Configuration backend was cuda but cuda is not available")

            device = torch.device("cuda:0") if config.get("device", "cuda") == "cuda" else torch.device("cpu")
            min_value = float(config.get("min_value", 0.0))
            max_value = float(config.get("max_value", 100.0))
            use_cache = True if config.get("use_cache", "True") == "True" else False
            method = config.get("method", "nn")
            scale = True if config.get("scale", "False") == "True" else False

        if use_cache and self.cache.contains(tilecolumn, tilerow, tilematrix):
            return self.cache.get(tilecolumn, tilerow, tilematrix)
        
        if tilematrix >= self.num_overviews:
            print("Tilematrix is greater than number of overviews.")
            return None

        # s = time.time()
        src = self.getshape(tilecolumn, tilerow, tilematrix).to(device).float()
        # e = time.time()

        # print("copy took {} seconds".format(e - s))
        
        src = src * self.scale + self.offset

        # print("mean is {}, max is {}, min is {}".format(src.mean(), src.max(), src.min()))

        # src = (src / 0.15).clamp_(min=0, max=len(self.cmap) - 1)
        src = src.clamp_(min=min_value, max=max_value)
        if not scale:
            src = src.div_(0.15).clamp_(min=0, max=len(self.cmap) - 1)

        if method == "nn":
            if scale:
                if src.min() != src.max():
                    src = src.sub_(src.min()).div_(src.max()).mul_(len(self.cmap) - 1)
                else:
                    src = src.sub_(src.min())
            
            src = src.long()
            image = self.cmap[src]
            low = image[:: 2 ** (self.num_overviews - tilematrix - 1), :: 2 ** (self.num_overviews - tilematrix - 1)]
        elif method == "avg":
            src = torch.nn.functional.avg_pool2d(src.unsqueeze(0), 2 ** (self.num_overviews - tilematrix - 1)).squeeze()
            if scale:
                if src.min() != src.max():
                    src = src.sub_(src.min()).div_(src.max()).mul_(len(self.cmap) - 1)
                else:
                    src = src.sub_(src.min())
                    
            src = src.long()
            low = self.cmap[src]
        else:
            raise ValueError("given downsampling method {} is not supported".format(method))

        print(low.shape, src.shape)


        # s2 = time.time()
        empty = torch.zeros((512, 512, 3), device=device, dtype=torch.uint8)

        # empty = torch.ByteTensor(512, 512, 3).fill_(0)
        empty[0:low.shape[0], 0:low.shape[1]] = low
        final = empty.cpu().numpy()
        # e2 = time.time()

        # print("copy back took {} seconds".format(e2 - s2))

        if use_cache:
            self.cache.store(tilecolumn, tilerow, tilematrix, final)

        return final