"""Segmentation: thresholds, ML pixel classification, object separation."""
from .threshold import hysteresis, multi_otsu, threshold  # noqa: F401
from .classifier import (PixelClassifier, auto_annotations, random_forest,  # noqa: F401
                         scribbles_from_json, slice_features)
from .objects import (connected_components, extract_class, remove_border_objects,  # noqa: F401
                      separate, smooth_labels)
