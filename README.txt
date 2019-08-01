# GPU-MRF Prototype

This prototype allows you to generate MRF files and imagery from raw data on-demand using GPUs. 

server.py is a simple flask server supporting a subset of the WMTS protocol. if run directly, it can be used to query tiles in a WMTS format. Mrfgen.py will generate an MRF from the specified data. A variety of configurations can be passed to these functions.
