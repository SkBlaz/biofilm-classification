"""Command-line preflight validation for image and feature inputs."""

from __future__ import annotations

import argparse
import json
import sys

from input_validation import (
    REPLICATION_UNITS,
    assess_replication_from_feature_table,
    validate_feature_table,
    validate_image_directory,
    write_json_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Folder containing .tif images")
    parser.add_argument("--features", help="Feature TSV to validate")
    parser.add_argument("--unlabelled", "--unlabeled", action="store_true")
    parser.add_argument("--replication-unit", choices=REPLICATION_UNITS)
    parser.add_argument("--nested-cv", action="store_true", help="Require nested CV for tuned learners")
    parser.add_argument("--output", help="Write the JSON report to this file")
    args = parser.parse_args()

    report = {"images": validate_image_directory(args.images, labelled=not args.unlabelled)}
    if args.features:
        report["features"] = validate_feature_table(args.features, require_label=not args.unlabelled)
    if args.features and args.replication_unit and not args.unlabelled:
        report["replication"] = assess_replication_from_feature_table(
            args.features,
            selected_unit=args.replication_unit,
            require_nested_cv=args.nested_cv,
        )
    report["ok"] = all(section.get("ok", False) for section in report.values())
    print(json.dumps(report, indent=2))
    if args.output:
        write_json_report(report, args.output)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
