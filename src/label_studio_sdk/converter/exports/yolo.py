import logging
import os
from label_studio_sdk.converter import brush
from label_studio_sdk.converter.utils import convert_annotation_to_yolo, convert_annotation_to_yolo_obb
from label_studio_sdk.converter.keypoints import build_kp_order

logger = logging.getLogger(__name__)

def process_keypoints_for_yolo(labels, label_path,
                               category_name_to_id, categories,
                               is_obb, kp_order):
    class_map = {c['name']: c['id'] for c in categories}

    rectangles = {}
    for item in labels:
        if item['type'].lower() == 'rectanglelabels':
            bbox_id = item['id']
            # Skip if there are no labels
            rect_labels = item.get('rectanglelabels') or []
            if not rect_labels:
                continue
            cls_name = rect_labels[0]
            cls_idx = class_map.get(cls_name)
            if cls_idx is None:
                continue

            x      = item['x'] / 100.0
            y      = item['y'] / 100.0
            width  = item['width']  / 100.0
            height = item['height'] / 100.0
            x_c    = x + width  / 2.0
            y_c    = y + height / 2.0

            rectangles[bbox_id] = {
                'class_idx': cls_idx,
                'x_center':  x_c,
                'y_center':  y_c,
                'width':     width,
                'height':    height,
                'kp_dict':   {}
            }

    for item in labels:
        if item['type'].lower() == 'keypointlabels':
            parent_id = item.get('parentID')
            if parent_id not in rectangles:
                continue
            kp_labels = item.get('keypointlabels') or []
            if not kp_labels:
                continue
            label_name = kp_labels[0]
            kp_x = item['x'] / 100.0
            kp_y = item['y'] / 100.0
            rectangles[parent_id]['kp_dict'][label_name] = (kp_x, kp_y, 2)  # 2 = visible

    lines = []
    for rect in rectangles.values():
        base = [
            rect['class_idx'],
            rect['x_center'],
            rect['y_center'],
            rect['width'],
            rect['height']
        ]
        keypoints = []
        for k in kp_order:
            keypoints.extend(rect['kp_dict'].get(k, (0.0, 0.0, 0)))
        line = ' '.join(map(str, base + keypoints))
        lines.append(line)

    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def process_and_save_yolo_annotations(labels, label_path, category_name_to_id, categories, is_obb, is_keypoints, label_config):
    if is_keypoints:
        kp_order = build_kp_order(label_config)
        process_keypoints_for_yolo(labels, label_path, category_name_to_id, categories, is_obb, kp_order)
        return categories, category_name_to_id

    # Stream annotations directly to a temporary file to avoid
    # accumulating them in memory and to preserve atomic writes.
    tmp_path = f"{label_path}.tmp"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for label in labels:
                category_names = []

                # Determine one or more classes assigned to this annotation.
                for key in (
                    "rectanglelabels",
                    "polygonlabels",
                    "brushlabels",
                    "labels",
                ):
                    if label.get(key):
                        category_names.extend(label[key])

                if not category_names:
                    logger.debug("Unknown label type or labels are empty: %s", label)
                    continue

                for category_name in category_names:
                    if category_name not in category_name_to_id:
                        category_id = len(categories)
                        category_name_to_id[category_name] = category_id
                        categories.append(
                            {"id": category_id, "name": category_name}
                        )
                    else:
                        category_id = category_name_to_id[category_name]

                    # Rectangle annotation
                    if (
                        "rectanglelabels" in label
                        or "rectangle" in label
                        or "labels" in label
                    ):
                        if is_obb:
                            obb_annotation = convert_annotation_to_yolo_obb(label)
                            if obb_annotation is None:
                                continue

                            top_left, top_right, bottom_right, bottom_left = (
                                obb_annotation
                            )
                            x1, y1 = top_left
                            x2, y2 = top_right
                            x3, y3 = bottom_right
                            x4, y4 = bottom_left

                            annotation_values = [
                                category_id,
                                x1,
                                y1,
                                x2,
                                y2,
                                x3,
                                y3,
                                x4,
                                y4,
                            ]
                        else:
                            annotation = convert_annotation_to_yolo(label)
                            if annotation is None:
                                continue

                            x, y, w, h = annotation
                            annotation_values = [category_id, x, y, w, h]

                    # Polygon annotation
                    elif "polygonlabels" in label or "polygon" in label:
                        points = label.get("points")
                        if not points:
                            continue

                        normalized_points = [
                            coordinate
                            for point_x, point_y in points
                            for coordinate in (point_x / 100.0, point_y / 100.0)
                        ]
                        annotation_values = [category_id, *normalized_points]

                    # Brush/mask annotation exported as a standard YOLO bbox.
                    elif (
                        "brushlabels" in label
                        or label.get("type", "").lower()
                        in ("brushlabels", "magicwand")
                    ):
                        if not brush.pycocotools_imported:
                            logger.warning(
                                "Skipping brush annotation because pycocotools "
                                "is not installed."
                            )
                            continue

                        rle = label.get("rle")
                        image_width = label.get("original_width")
                        image_height = label.get("original_height")

                        if not rle or not image_width or not image_height:
                            logger.warning(
                                "Skipping brush annotation without RLE or image "
                                "dimensions: %s",
                                label,
                            )
                            continue

                        coco_rle = brush.ls_rle_to_coco_rle(
                            rle,
                            image_height,
                            image_width,
                        )
                        x_min, y_min, box_width, box_height = (
                            brush.get_cocomask_bounding_box(coco_rle)
                        )

                        if box_width <= 0 or box_height <= 0:
                            logger.warning(
                                "Skipping brush annotation with an empty mask: %s",
                                label,
                            )
                            continue

                        annotation_values = [
                            category_id,
                            (x_min + box_width / 2.0) / image_width,
                            (y_min + box_height / 2.0) / image_height,
                            box_width / image_width,
                            box_height / image_height,
                        ]

                    else:
                        logger.warning(
                            "Unsupported YOLO annotation type: %s",
                            label,
                        )
                        continue

                    f.write(" ".join(map(str, annotation_values)) + "\n")

        os.replace(tmp_path, label_path)

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return categories, category_name_to_id