from core import Product

name = "GHRSST_L4_MUR_Sea_Surface_Temperature"
product = Product(name, offset = 25.0, scale = 0.001, device="cuda")

product.mrfgen({"format" : "tiff", "device" : "cuda", "method" : "avg", "output_dir" : "output"})