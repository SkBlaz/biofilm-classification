"""Shared segmentation settings, calibrated masks, and TIFF conversion."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

try:
    from .feature_metrics import (
        bem_threshold,
        multi_otsu_labels,
        multi_otsu_thresholds,
        otsu_threshold,
        robust_background_threshold,
    )
except ImportError:
    from feature_metrics import bem_threshold, multi_otsu_labels, multi_otsu_thresholds, otsu_threshold, robust_background_threshold

GROUPS = ("basic", "objects", "advanced", "intensity", "geometry", "serial")
ENV_SETTINGS = {
    "segmentation_approach": "IMAGINE_SEGMENTATION_APPROACH",
    "threshold_method": "IMAGINE_THRESHOLD_METHOD",
    "threshold_scale": "IMAGINE_THRESHOLD_SCALE",
    "threshold_value": "IMAGINE_THRESHOLD_VALUE",
    "threshold_sensitivity": "IMAGINE_THRESHOLD_SENSITIVITY",
    "bem_tolerance": "IMAGINE_BEM_TOLERANCE",
    "dim_class_assignment": "IMAGINE_DIM_CLASS_ASSIGNMENT",
    "image_dimension": "IMAGINE_IMAGE_DIMENSION",
    "feature_groups": "IMAGINE_FEATURE_GROUPS",
    "conversion_max": "IMAGINE_CONVERSION_MAX",
    "masks_dir": "IMAGINE_MASKS_DIR",
    "connectivity_3d": "IMAGINE_CONNECTIVITY_3D",
    "minimum_object_area_um2": "IMAGINE_MIN_OBJECT_AREA_UM2",
    "minimum_object_volume_um3": "IMAGINE_MIN_OBJECT_VOLUME_UM3",
    "local_density_radius_um": "IMAGINE_LOCAL_DENSITY_RADIUS_UM",
    "qc_only": "IMAGINE_QC_ONLY",
}


def environment_settings(environ=None):
    environ = os.environ if environ is None else environ
    return {key: environ[name] for key, name in ENV_SETTINGS.items() if environ.get(name, "") != ""}


def normalize_settings(raw):
    """Validate GUI and CLI settings identically."""
    method = str(raw.get("threshold_method", "otsu"))
    value = raw.get("threshold_value")
    value = None if value in (None, "") else value
    approach = raw.get("segmentation_approach") or ("manual" if method == "manual" or value is not None else "automatic")
    if approach not in {"automatic", "manual", "import"}:
        raise ValueError("Choose automatic, manual, or imported-mask segmentation")
    if method not in {"otsu", "multi_otsu", "robust_background", "bem", "manual"}:
        raise ValueError("Choose a supported threshold method")
    if approach == "automatic" and (method == "manual" or value is not None):
        raise ValueError("Choose Manual / tuned threshold to apply the entered cutoff")
    scale = str(raw.get("threshold_scale", "uint8"))
    scale = "uint8" if scale in {"raw", "raw_intensity", "raw_uint8"} else scale
    if scale not in {"uint8", "stack_normalized"}:
        raise ValueError("Choose 8-bit intensity or stack-normalized threshold units")
    if approach == "manual":
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Enter a manual threshold value") from exc
        maximum = 255 if scale == "uint8" else 1
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"Manual threshold must be between 0 and {maximum}")
    elif approach == "import":
        value = None
    dimension = str(raw.get("image_dimension", "auto")).lower()
    if dimension not in {"auto", "2d", "3d"}:
        raise ValueError("Choose 2D or 3D images")
    groups = raw.get("feature_groups", ["all"])
    if isinstance(groups, str):
        groups = groups.split(",")
    allowed_groups = (*GROUPS, "all", "2d") if dimension == "2d" else (*GROUPS, "all")
    if not isinstance(groups, (list, tuple)) or not groups or any(group not in allowed_groups for group in groups):
        raise ValueError("Select at least one supported feature group")
    groups = list(GROUPS) if "all" in groups else [group for group in GROUPS if group in groups]
    if dimension == "2d":
        groups = ["2d", "intensity", "geometry"]
    settings = {
        "segmentation_approach": approach,
        "threshold_method": method,
        "threshold_boundary_mode": "comstat2",
        "threshold_scale": scale,
        "threshold_value": value,
        "dim_class_assignment": str(raw.get("dim_class_assignment", "foreground")),
        "image_dimension": dimension,
        "feature_groups": groups,
        "masks_dir": str(raw.get("masks_dir", "")),
        "qc_only": raw.get("qc_only", False) in (True, "true", "1", 1),
    }
    if approach == "import" and not settings["masks_dir"]:
        raise ValueError("Provide a directory containing matching TIFF masks")
    if settings["dim_class_assignment"] not in {"foreground", "background"}:
        raise ValueError("Choose foreground or background for the dim class")
    try:
        connectivity = float(raw.get("connectivity_3d", 26))
    except (TypeError, ValueError) as exc:
        raise ValueError("3D connectivity must be 6, 18, or 26") from exc
    if connectivity not in {6, 18, 26}:
        raise ValueError("3D connectivity must be 6, 18, or 26")
    settings["connectivity_3d"] = int(connectivity)
    for key, default in (
        ("threshold_sensitivity", 1.0),
        ("bem_tolerance", 0.10),
        ("minimum_object_area_um2", 0.0),
        ("minimum_object_volume_um3", 0.0),
        ("local_density_radius_um", 2.0),
        ("conversion_max", None),
    ):
        item = raw.get(key, default)
        if key == "conversion_max" and item in (None, ""):
            settings[key] = None
            continue
        try:
            item = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key.replace('_', ' ')} must be numeric") from exc
        positive = key not in {"minimum_object_area_um2", "minimum_object_volume_um3"}
        if not math.isfinite(item) or item < 0 or (positive and item == 0):
            raise ValueError(f"{key.replace('_', ' ')} must be finite and {'positive' if positive else 'non-negative'}")
        if key == "bem_tolerance" and item >= 1:
            raise ValueError("BEM slope-change tolerance must be between 0 and 1")
        settings[key] = item
    return settings


def tiff_info(path):
    """Read dimensions without loading pixel data."""
    with tifffile.TiffFile(path) as image:
        series = image.series[0]
        shape = series.shape
        axes = series.axes
        if axes == "YX":
            shape = (1, *shape)
        elif axes not in {"ZYX", "QYX", "IYX"}:
            raise ValueError(f"{Path(path).name}: expected a grayscale YX or ZYX TIFF; found axes {axes}. Export one channel/timepoint.")
        if len(shape) != 3 or any(size < 1 for size in shape):
            raise ValueError("TIFF must contain a non-empty 2D image or 3D stack")
        significant_bits = None
        if image.ome_metadata:
            root = ET.fromstring(image.ome_metadata)
            for element in root.iter():
                if element.tag.endswith("Pixels") and element.get("SignificantBits"):
                    significant_bits = int(element.get("SignificantBits"))
                    break
        return tuple(shape), significant_bits


def read_tiff_stack(path):
    shape, significant_bits = tiff_info(path)
    stack = tifffile.imread(path).reshape(shape)
    if not np.isfinite(stack).all():
        raise ValueError("Image intensities must be finite")
    return stack, significant_bits


def to_uint8(stack, conversion_max=None, significant_bits=None):
    """Scale once per stack, with round-half-up quantization."""
    values = np.asarray(stack)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Image must be non-empty and finite")
    if values.dtype == np.uint8 and conversion_max is None:
        return values.copy(), {"rule": "identity", "lower": 0.0, "upper": 255.0, "source_dtype": "uint8", "clipped_fraction": 0.0}
    lower = min(0.0, float(values.min()))
    if conversion_max is not None:
        upper, source = float(conversion_max), "explicit acquisition maximum"
    elif significant_bits is not None:
        if not 1 <= significant_bits <= 32:
            raise ValueError("TIFF SignificantBits must be between 1 and 32")
        upper, source = float(2**significant_bits - 1), "OME SignificantBits"
    elif np.issubdtype(values.dtype, np.integer):
        upper, source = float(np.iinfo(values.dtype).max), "storage dtype range"
    else:
        upper, source = float(values.max()), "whole-stack observed range"
    if not math.isfinite(upper) or upper < lower:
        raise ValueError("Conversion maximum must be finite and exceed the lower bound")
    if upper == lower:
        output = np.zeros_like(values, dtype=np.uint8)
    else:
        output = np.floor(np.clip((values.astype(float) - lower) / (upper - lower), 0, 1) * 255 + 0.5).astype(np.uint8)
    metadata = {
        "rule": "floor(255 * clip((raw - lower) / (upper - lower), 0, 1) + 0.5)",
        "lower": lower,
        "upper": upper,
        "bounds_source": source,
        "source_dtype": str(values.dtype),
        "significant_bits": significant_bits,
        "clipped_fraction": float(np.mean((values < lower) | (values > upper))),
    }
    return output, metadata


def mask_for_image(filepath, masks_dir):
    """Pair exact stems, optionally ending in _mask."""
    stem = Path(filepath).stem
    directory = Path(masks_dir)
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and path.stem in {stem, stem + "_mask"}
    )
    if len(matches) != 1:
        raise ValueError(f"{Path(filepath).name}: upload exactly one matching mask named {stem}.tif or {stem}_mask.tif")
    return matches[0]


def read_mask(path, shape):
    labels, _ = read_tiff_stack(path)
    if labels.shape != shape:
        raise ValueError(f"{Path(path).name}: mask dimensions {labels.shape} differ from image {shape}")
    if np.min(labels) < 0:
        raise ValueError("Mask labels must be non-negative")
    if np.issubdtype(labels.dtype, np.floating):
        return (labels != 0).astype(np.int32), False
    if not np.issubdtype(labels.dtype, np.integer) and labels.dtype != bool:
        raise ValueError("Masks must contain integer labels or numeric binary values")
    unique = np.unique(labels)
    labelled = len(unique[unique > 0]) > 1
    if labelled:
        # Compact arbitrary labels without losing object identities.
        _, inverse = np.unique(labels, return_inverse=True)
        if unique[0] != 0:
            inverse += 1
        labels = inverse.reshape(shape).astype(np.int32)
    else:
        labels = (labels > 0).astype(np.int32)
    return labels, labelled


def filter_mask(mask, dimensions, settings, labels=None):
    """Remove small objects before all mask measurements."""
    is_2d = mask.shape[0] == 1
    image = mask[0] if is_2d else mask
    if labels is None:
        rank = 2 if is_2d else {6: 1, 18: 2, 26: 3}[settings["connectivity_3d"]]
        objects, _ = ndimage.label(image, ndimage.generate_binary_structure(image.ndim, rank))
    else:
        objects = labels[0] if is_2d else labels.copy()
    counts = np.bincount(objects.ravel())
    size = np.prod(dimensions[:2]) if is_2d else np.prod(dimensions)
    minimum = settings["minimum_object_area_um2" if is_2d else "minimum_object_volume_um3"]
    keep = counts * size >= minimum
    keep[0] = False
    objects = np.where(keep[objects], objects, 0)
    if is_2d:
        objects = objects[np.newaxis]
    return objects > 0, objects


def generate_mask(image8, settings, imported=None):
    """Return the exact applied threshold and unfiltered mask."""
    approach = settings["segmentation_approach"]
    result = {"method": settings["threshold_method"], "threshold_uint8": None, "class_thresholds_uint8": None}
    if approach == "import":
        if imported is None:
            raise ValueError("Import masks requires a matching TIFF mask")
        result["method"] = "import_mask"
        return imported > 0, result
    classes = None
    if approach == "manual":
        cutoff = settings["threshold_value"] * (255 if settings["threshold_scale"] == "stack_normalized" else 1)
        result["method"] = "manual"
    else:
        sensitivity = settings["threshold_sensitivity"]
        method = settings["threshold_method"]
        if method == "otsu":
            # Histogram Otsu places the last background bin at k.
            cutoff = (otsu_threshold(image8) + 1) * sensitivity
        elif method == "robust_background":
            cutoff = robust_background_threshold(image8) * sensitivity
        elif method == "bem":
            cutoff = bem_threshold(image8, settings["bem_tolerance"])
        elif method == "multi_otsu":
            classes = tuple(value * sensitivity for value in multi_otsu_thresholds(image8))
            cutoff = classes[0 if settings["dim_class_assignment"] == "foreground" else 1]
        else:
            raise ValueError("Automatic segmentation requires an automatic method")
    result.update(threshold_uint8=float(cutoff), class_thresholds_uint8=classes)
    if classes:
        result["class_labels"] = multi_otsu_labels(image8, *classes)
    return image8 >= cutoff, result
