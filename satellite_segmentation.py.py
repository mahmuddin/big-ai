import os
import torch
import rasterio
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_geospatial import tms_to_geotiff

# Konfigurasi
bbox = [-95.3704, 29.6762, -95.368, 29.6775]  # Koordinat area
image = 'satellite.tif'

# Download citra satelit
tms_to_geotiff(output=image, bbox=bbox, zoom=20, source='Satellite')

# Load model SAM
out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
checkpoint = os.path.join(out_dir, 'sam_vit_h_4b8939.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Load citra satelit
with rasterio.open(image) as src:
    image_array = src.read().transpose(1, 2, 0)  # Ubah format array

# Generate segmentasi
masks = mask_generator.generate(image_array)

# Simpan hasil segmentasi
mask_output = 'segment.tif'
with rasterio.open(mask_output, 'w', **src.profile) as dst:
    dst.write(masks, 1)

print("Segmentasi selesai! Hasil disimpan di:", mask_output)
