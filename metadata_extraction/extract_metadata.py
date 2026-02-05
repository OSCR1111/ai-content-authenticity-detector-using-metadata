from PIL import Image
from PIL.ExifTags import TAGS

SELECTED_TAGS = [
    "Make",
    "Model",
    "Software",
    "DateTime",
    "ExifImageWidth",
    "ExifImageHeight",
    "Orientation",
    "ColorSpace"
]

def extract_metadata(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    metadata = {}

    # Initialize all fields as missing
    for tag in SELECTED_TAGS:
        metadata[tag] = None

    if exif_data:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag in SELECTED_TAGS:
                metadata[tag] = value

    return metadata