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
    roles = {"first": 0, "middle": mask.shape[0] // 2, "top": mask.shape[0] - 1}
    artifacts = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, (role, index) in enumerate(roles.items()):
        overlay = np.ma.masked_where(~mask[index], mask[index])
        axes[0, column].imshow(image8[index], cmap="gray", vmin=0, vmax=255)
        axes[0, column].imshow(overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        axes[1, column].imshow(mask[index], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"{role.title()} slice: z={index}")
        axes[1, column].set_title("Applied biomass mask")
        for row in (0, 1):
            axes[row, column].axis("off")
        overlay_name = f"{stem}_overlay_{role}.png"
        mask_name = f"{stem}_mask_{role}.png"
        plt.imsave(directory / mask_name, mask[index].astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
        view, axis = plt.subplots(figsize=(5, 5))
        axis.imshow(image8[index], cmap="gray", vmin=0, vmax=255)
        axis.imshow(overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        axis.set_title(f"{role.title()} slice: z={index}")
        axis.axis("off")
        view.tight_layout()
        view.savefig(directory / overlay_name, dpi=140)
        plt.close(view)
        artifacts[role] = {"slice_index": index, "overlay": overlay_name, "mask": mask_name}
    overlay_name = f"{stem}_segmentation_qc.png"
    fig.tight_layout()
    fig.savefig(directory / overlay_name, dpi=160)
    plt.close(fig)

    cutoff = result["threshold_uint8"]
    histogram = np.bincount(image8.ravel(), minlength=256)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(np.arange(256), histogram, width=1, color="0.35")
    if cutoff is not None:
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
    border = np.any((coordinates == 0) | (coordinates == np.asarray(mask.shape) - 1), axis=1)
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
        "bem_tolerance": settings["bem_tolerance"] if result["method"] == "bem" else None,
        "dim_class_assignment": settings["dim_class_assignment"] if result["method"] == "multi_otsu" else None,
        "conversion": conversion,
        "thresholding_image": converted_path.name,
        "imported_mask": Path(mask_path).name if mask_path else None,
        "mask": mask_output.name,
        "object_labels": label_output.name,
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
        "slice_qc": artifacts,
        "histogram": histogram_name,
        "threshold_curve": curve_name,
    }
    (directory / f"{stem}_segmentation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
