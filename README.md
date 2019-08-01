# GPU-Data-To-Image Prototype

This prototype allows you to quickly generate image tiles and MRF files from raw data and on-demand using GPUs, using
different colormaps and formats.

## Setup

The easiest way to use this software is to download a data file like `20160628-JPL-L4UHfnd-GLOB-v01-fv04-MUR.nc` and drag it to the data folder in this repository. You can run `pip install -r requirements.txt` to automatically install Python dependencies. Note that you must have a computer with an NVIDIA GPU and CUDA must be installed (follow the instructions at https://developer.nvidia.com/cuda-downloads. Then you can run a simple tile generation server with `python server.py`, or launch the WMTS-like flask server with `./run.sh`. You can also use the `parser.py` file to pull date from the public NEXUS/SDAP backend.

## core.py

core.py contains the core algorithms for generating mrf files and producing tiles from data. This has been tested mostly
on a binary data file 20160628-JPL-L4UHfnd-GLOB-v01-fv04-MUR.nc, but should work with minor adjustments for other data
files and data pulled directly from the NEXUS backend. In particular, you can preload tiles from the NEXUS backend and save
them in a pickle format or use the data directly. See parser.py for this utility.

## server.py

server.py is a simple flask server supporting a subset of the WMTS protocol. If run directly, it can be used to query 
tiles in a WMTS format. This is very primitive, and the WMTS support extends only to the TileMatrix, TileCol, and TileRow
fields in the WMTS string, as well as any keywords you define in server.py. To start the server, use `./run.sh`, which will
run a local server at localhost:5000 by default.

## mrfgen.py

mrfgen.py generates an MRF with desired configurations from the raw data. See core.py for configuration options supported.

## parser.py

parser.py loads data from the NEXUS JSON backend and formats it into a Numpy array. This is largely unfinished and will need
some tweaking. Hopefully when this is incorporated directly into the backend we will receive Numpy arrays directly instead
of JSON files. 

## Algorithm Overview

This code uses PyTorch for GPU acceleration, a near Numpy-clone with built-in GPU support for PyTorch Tensors, equivalents
of Numpy arrays. You can simply call my_tensor.to("cuda:0") or my_tensor.to("cpu") to make all code run on the GPU or 
CPU. Basically, all we do is download and parse colormaps from GIBS or the Sea Level Change portal, load the raw data
into a PyTorch tensor, apply some appropriate scalings (some netcdf files are stored as long integers with a floating
point offset specified), rescale to integers in the range [0, len(colormap)), convert to RGB colors, copy back to the CPU,
and server to the user. Downsampling is accomplished either by simply indexing the arrays with arr[::2, ::2] (for 2x downsampling),
or using torch.nn.functional.avg_pool2d (which does average downsampling). This second method can be replicated in Numpy
using PyTorch, Numpy convolutions, or skimage.measure.block_reduce. 

This can be run on the cpu by passing device="cpu" or the appropriate dictionary config to mrfgen or gettile. 

To adapt this to the Apache Spark backend, basically don't load data in the Product constructor and dispatch Apache
Spark workers to handle a subset of the data tile requested by the user. For downsampled layers, you should take every
2 ** n column, i.e. use arr[::2] to downsample, or do average downsampling, and then apply the appropriate colormap
and encode the result as a PNG or JPEG (faster) and serve it back to the user with an image ContentType.

The current implementation assumes that colormaps are evenly spaced, so just dividing by the bin spacing works 
for binning. This isn't always true, but you can probably use a dictionary or numpy digitize function for more 
exotic colormaps. 

The algorithm only loads the data needed for each tile, which is the entire dataset for the level 0 TileMatrix, but drops by a
factor of 4 for each lower-level overview. 

## MRF Generation

MRF generation uses the same basic algorithm, but loads the entire dataset and segments it into all possible tiles which are
saved in the MRF format. This can use various image formats within the MRF and different downsampling methods. This could be
significantly improved by parallelizing the process of encoding the image files, since the main overhead is not computation
but the expense of creating PNG or JPEG strings from the Numpy arrays. 
