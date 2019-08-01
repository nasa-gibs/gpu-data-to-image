from core import Product

name = "AVHRR_OI_L4_GHRSST_NCEI" # name = "GHRSST_L4_MUR_Sea_Surface_Temperature"
product = Product(name, device="cuda") #  offset = 25.0, scale = 0.001, 

product.mrfgen({"format" : "tiff", "device" : "cuda", "method" : "avg", "output_dir" : "output"})