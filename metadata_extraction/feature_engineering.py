import pandas as pd

def metadata_to_features(metadata, label):
    features = {
        "has_make": int(metadata["Make"] is not None),
        "has_model": int(metadata["Model"] is not None),
        "has_software": int(metadata["Software"] is not None),
        "has_datetime": int(metadata["DateTime"] is not None),
        "has_orientation": int(metadata["Orientation"] is not None),
        "has_colorspace": int(metadata["ColorSpace"] is not None),
        "width": metadata["ExifImageWidth"] or 0,
        "height": metadata["ExifImageHeight"] or 0,
        "label": label
    }
    return features
