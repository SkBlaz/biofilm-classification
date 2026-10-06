"""QC artifacts generated from the exact thresholding image and mask."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def write_qc(filepath, outfolder, image8, mask, labels, settings, result, conversion, dimensions, mask_path=None):
    directory = Path(outfolder)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem
    converted_path = directory / f"{stem}_thresholding_uint8.tif"
    mask_output = directory / f"{stem}_mask.tif"
    label_output = directory / f"{stem}_object_labels.tif"
    for path, data in ((converted_path, image8), (mask_output, mask.astype(np.uint8) * 255), (label_output, labels.astype(np.uint32))):
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "ZYX"})
    overlay_output = directory / f"{stem}_overlay.tif"
    color_output = directory / f"{stem}_object_labels_rgb.tif"
    overlay = np.repeat(image8[..., np.newaxis], 3, axis=-1)
    overlay[mask] = np.round(0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    identities, inverse = np.unique(labels, return_inverse=True)
    colors = np.random.default_rng(0).integers(64, 256, (len(identities), 3), dtype=np.uint8)
    colors[identities == 0] = 0
    colored_labels = colors[inverse].reshape((*labels.shape, 3))
    for path, data in ((overlay_output, overlay), (color_output, colored_labels)):
        tifffile.imwrite(path, data, photometric="rgb", metadata={"axes": "ZYXS"})
    overlay_name = f"{stem}_segmentation_qc.png"
    plt.imsave(directory / overlay_name, overlay.max(axis=0))

    cutoff = result["threshold_uint8"]
    base = result.get("base_threshold_uint8")
    histogram = np.bincount(image8.ravel(), minlength=256)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(np.arange(256), histogram, width=1, color="0.35")
    if cutoff is not None:
        if base is not None:
            axis.axvline(base, color="blue", linestyle=":", label=f"Base automatic threshold: {base:.5g}")
        axis.axvline(cutoff, color="red", linestyle="--", label=f"Applied threshold: {cutoff:.5g}")
        for value in result["class_thresholds_uint8"] or ():
            axis.axvline(value, color="blue", linestyle=":")
        axis.legend()
    axis.set_xlabel("8-bit intensity (ImageJ, 0–255)")
    axis.set_ylabel("Voxel count")
    histogram_name = f"{stem}_histogram.png"
    fig.tight_layout()
    fig.savefig(directory / histogram_name, dpi=160)
    plt.close(fig)
    curve_name = None
    if cutoff is not None:
        fractions = np.cumsum(histogram[::-1])[::-1] / image8.size
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(np.arange(256), fractions)
        if base is not None:
            axis.axvline(base, color="blue", linestyle=":", label=f"Base automatic threshold: {base:.5g}")
        axis.axvline(cutoff, color="red", linestyle="--", label=f"Applied threshold: {cutoff:.5g}")
        axis.scatter([cutoff], [mask.mean()], color="red", label="Applied mask after size filtering")
        axis.set_xlabel("8-bit threshold (COMSTAT2: intensity ≥ cutoff)")
        axis.set_ylabel("Biomass fraction")
        axis.legend()
        curve_name = f"{stem}_threshold_curve.png"
        fig.tight_layout()
        fig.savefig(directory / curve_name, dpi=160)
        plt.close(fig)
    coordinates = np.argwhere(mask)
    active_axes = np.array([1, 2]) if mask.shape[0] == 1 else np.array([0, 1, 2])
    border = np.any((coordinates[:, active_axes] == 0) | (coordinates[:, active_axes] == np.asarray(mask.shape)[active_axes] - 1), axis=1)
    classes = result.get("class_labels")
    record = {
        "image": Path(filepath).name,
        "segmentation_version": 2,
        "segmentation_approach": settings["segmentation_approach"],
        "requested_method": settings["threshold_method"],
        "method": result["method"],
        "submitted_threshold_value": settings["threshold_value"],
        "submitted_threshold_units": settings["threshold_scale"] if settings["segmentation_approach"] == "manual" else None,
        "selected_threshold_value": (settings["threshold_value"] if settings["segmentation_approach"] == "manual" else cutoff),
        "selected_threshold_units": (settings["threshold_scale"] if settings["segmentation_approach"] == "manual" else "uint8")
        if cutoff is not None
        else None,
        "threshold_uint8": cutoff,
        "base_threshold_uint8": base,
        "threshold_stack_normalized": cutoff / 255 if cutoff is not None else None,
        "class_thresholds_uint8": result["class_thresholds_uint8"],
        "class_thresholds_stack_normalized": [value / 255 for value in result["class_thresholds_uint8"]]
        if result["class_thresholds_uint8"]
        else None,
        "threshold_boundary_mode": "comstat2",
        "threshold_sensitivity": settings["threshold_sensitivity"] if settings["segmentation_approach"] == "automatic" else None,
        "configured_threshold_sensitivity": settings["threshold_sensitivity"],
        "configured_bem_tolerance": settings["bem_tolerance"],
        "configured_conversion_max": settings["conversion_max"],
        "configured_conversion_range_mode": settings["conversion_range_mode"],
        "bem_tolerance": settings["bem_tolerance"] if result["method"] == "bem" else None,
        "dim_class_assignment": settings["dim_class_assignment"] if result["method"] == "multi_otsu" else None,
        "conversion": conversion,
        "thresholding_image": converted_path.name,
        "imported_mask": Path(mask_path).name if mask_path else None,
        "mask": mask_output.name,
        "object_labels": label_output.name,
        "object_labels_rgb": color_output.name,
        "overlay": overlay_output.name,
        "voxel_size_um": dict(zip(("x", "y", "z"), dimensions)),
        "connectivity_2d": 8,
        "connectivity_3d": settings["connectivity_3d"],
        "minimum_object_area_um2": settings["minimum_object_area_um2"],
        "minimum_object_volume_um3": settings["minimum_object_volume_um3"],
        "local_density_radius_um": settings["local_density_radius_um"],
        "array_axis_order": ["z", "y", "x"],
        "image_dimension": "2d" if mask.shape[0] == 1 else "3d",
        "feature_groups": settings["feature_groups"],
        "skipped_feature_groups": ["3D object features: image has one slice"] if mask.shape[0] == 1 else [],
        "no_biomass_detected": not bool(mask.any()),
        "foreground_fraction": float(mask.mean()),
        "foreground_touching_image_border_fraction": float(border.mean()) if border.size else None,
        "top_layer_occupancy_percent": float(mask[-1].mean() * 100),
        "saturation_fraction": float(np.mean(image8 == 255)),
        "three_class_fractions": {
            "background": float(np.mean(classes == 0)),
            "dim-intensity": float(np.mean(classes == 1)),
            "bright-intensity": float(np.mean(classes == 2)),
        }
        if classes is not None
        else None,
        "qc_overlay": overlay_name,
        "foreground_touching_image_border_definition": "foreground voxels on image edges divided by all foreground voxels; Z excluded for 2D",
        "histogram": histogram_name,
        "threshold_curve": curve_name,
    }
    (directory / f"{stem}_segmentation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
