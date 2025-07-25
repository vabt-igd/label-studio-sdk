import argparse
import datetime
import hashlib
import io
import logging
import math
import os
import re
import shutil
import urllib
import wave
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from lxml import etree
from nltk.tokenize.treebank import TreebankWordTokenizer

from label_studio_sdk._extensions.label_studio_tools.core.utils.params import get_env
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import safe_build_path

logger = logging.getLogger(__name__)

_LABEL_TAGS = {"Label", "Choice"}
_NOT_CONTROL_TAGS = {
    "Filter",
}
LOCAL_FILES_DOCUMENT_ROOT = get_env(
    "LOCAL_FILES_DOCUMENT_ROOT", default=os.path.abspath(os.sep)
)

TreebankWordTokenizer.PUNCTUATION = [
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (re.compile(r"\.\.\."), r" ... "),
    (re.compile(r"[;@#$/%&]"), r" \g<0> "),
    (
        re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
        r"\1 \2\3 ",
    ),  # Handles the final period.
    (re.compile(r"[?!]"), r" \g<0> "),
    (re.compile(r"([^'])' "), r"\1 ' "),
]


class ExpandFullPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def tokenize(text):
    tok_start = 0
    out = []
    for tok in text.split():
        if len(tok):
            out.append((tok, tok_start))
            tok_start += len(tok) + 1
        else:
            tok_start += 1
    return out


def create_tokens_and_tags(text, spans):
    # tokens_and_idx = tokenize(text) # This function doesn't work properly if text contains multiple whitespaces...
    token_index_tuples = [
        token for token in TreebankWordTokenizer().span_tokenize(text)
    ]
    tokens_and_idx = [(text[start:end], start) for start, end in token_index_tuples]
    if spans and all(
        [
            span.get("start") is not None and span.get("end") is not None
            for span in spans
        ]
    ):
        spans = list(sorted(spans, key=itemgetter("start")))
        span = spans.pop(0)
        span_start = span["start"]
        span_end = span["end"] - 1
        prefix = "B-"
        tokens, tags = [], []
        for token, token_start in tokens_and_idx:
            tokens.append(token)
            token_end = (
                token_start + len(token) - 1
            )  # "- 1" - This substraction is wrong. token already uses the index E.g. "Hello" is 0-4
            token_start_ind = token_start  # It seems like the token start is too early.. for whichever reason

            # if for some reason end of span is missed.. pop the new span (Which is quite probable due to this method)
            # Attention it seems like span['end'] is the index of first char afterwards. In case the whitespace is part of the
            # labell we need to subtract one. Otherwise next token won't trigger the span update.. only the token after next..
            if token_start_ind > span_end:
                while spans:
                    span = spans.pop(0)
                    span_start = span["start"]
                    span_end = span["end"] - 1
                    prefix = "B-"
                    if token_start <= span_end:
                        break
            # Add tag "O" for spans that:
            # - are empty
            # - span start has passed over token_end
            # - do not have any label (None or empty list)
            if not span or token_end < span_start or not span.get("labels"):
                tags.append("O")
            elif span_start <= token_end and span_end >= token_start_ind:
                tags.append(prefix + span["labels"][0])
                prefix = "I-"
            else:
                tags.append("O")
    else:
        tokens = [token for token, _ in tokens_and_idx]
        tags = ["O"] * len(tokens)

    return tokens, tags


def _get_upload_dir(project_dir=None, upload_dir=None):
    """Return either upload_dir, or path by LS_UPLOAD_DIR, or project_dir/upload"""
    if upload_dir:
        return upload_dir
    upload_dir = os.environ.get("LS_UPLOAD_DIR")
    if not upload_dir and project_dir:
        upload_dir = os.path.join(project_dir, "upload")
        if not os.path.exists(upload_dir):
            upload_dir = None
    if not upload_dir:
        raise FileNotFoundError(
            "Can't find upload dir: either LS_UPLOAD_DIR or project should be passed to converter"
        )
    return upload_dir


def download(
    url,
    output_dir,
    filename=None,
    project_dir=None,
    return_relative_path=False,
    upload_dir=None,
    download_resources=True,
):
    is_local_file = url.startswith("/data/") and "?d=" in url
    is_uploaded_file = url.startswith("/data/upload")

    if is_uploaded_file:
        upload_dir = _get_upload_dir(project_dir, upload_dir)
        filename = urllib.parse.unquote(url.replace("/data/upload/", ""))
        filepath = safe_build_path(upload_dir, filename)
        logger.debug(
            f"Copy {filepath} to {output_dir}".format(
                filepath=filepath, output_dir=output_dir
            )
        )
        if download_resources:
            shutil.copy(filepath, output_dir)
        if return_relative_path:
            return os.path.join(
                os.path.basename(output_dir), os.path.basename(filename)
            )
        return filepath

    if is_local_file:
        filename, dir_path = url.split("/data/", 1)[-1].split("?d=")
        dir_path = str(urllib.parse.unquote(dir_path))
        filepath = safe_build_path(LOCAL_FILES_DOCUMENT_ROOT, dir_path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        if download_resources:
            shutil.copy(filepath, output_dir)
        return filepath

    if filename is None:
        basename, ext = os.path.splitext(os.path.basename(urlparse(url).path))
        filename = f"{basename}{ext}"
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            filename = (
                basename
                + "_"
                + hashlib.md5(
                    url.encode() + str(datetime.datetime.now().timestamp()).encode()
                ).hexdigest()[:4]
                + ext
            )

    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        logger.info("Download {url} to {filepath}".format(url=url, filepath=filepath))
        if download_resources:
            r = requests.get(url)
            r.raise_for_status()
            with io.open(filepath, mode="wb") as fout:
                fout.write(r.content)
    if return_relative_path:
        return os.path.join(os.path.basename(output_dir), os.path.basename(filename))
    return filepath


def get_image_size(image_path):
    return Image.open(image_path).size


def get_image_size_and_channels(image_path):
    i = Image.open(image_path)
    w, h = i.size
    c = len(i.getbands())
    return w, h, c


def get_audio_duration(audio_path):
    with wave.open(audio_path, mode="r") as f:
        return f.getnframes() / float(f.getframerate())


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_config(config_string):
    """
    :param config_string: Label config string
    :return: structured config of the form:
    {
        "<ControlTag>.name": {
            "type": "ControlTag",
            "to_name": ["<ObjectTag1>.name", "<ObjectTag2>.name"],
            "inputs: [
                {"type": "ObjectTag1", "value": "<ObjectTag1>.value"},
                {"type": "ObjectTag2", "value": "<ObjectTag2>.value"}
            ],
            "labels": ["Label1", "Label2", "Label3"] // taken from "alias" if exists or "value"
    }
    """
    if not config_string:
        return {}

    def _is_input_tag(tag):
        return tag.attrib.get("name") and tag.attrib.get("value")

    def _is_output_tag(tag):
        return (
            tag.attrib.get("name")
            and tag.attrib.get("toName")
            and tag.tag not in _NOT_CONTROL_TAGS
        )

    def _get_parent_output_tag_name(tag, outputs):
        # Find parental <Choices> tag for nested tags like <Choices><View><View><Choice>...
        parent = tag
        while True:
            parent = parent.getparent()
            if parent is None:
                return
            name = parent.attrib.get("name")
            if name in outputs:
                return name

    try:
        xml_tree = etree.fromstring(config_string)
    except etree.XMLSyntaxError as e:
        raise ValueError(str(e))

    inputs, outputs, labels = {}, {}, defaultdict(dict)
    for tag in xml_tree.iter():
        if _is_output_tag(tag):
            tag_info = {"type": tag.tag, "to_name": tag.attrib["toName"].split(",")}
            # Grab conditionals if any
            conditionals = {}
            if tag.attrib.get("perRegion") == "true":
                if tag.attrib.get("whenTagName"):
                    conditionals = {"type": "tag", "name": tag.attrib["whenTagName"]}
                elif tag.attrib.get("whenLabelValue"):
                    conditionals = {
                        "type": "label",
                        "name": tag.attrib["whenLabelValue"],
                    }
                elif tag.attrib.get("whenChoiceValue"):
                    conditionals = {
                        "type": "choice",
                        "name": tag.attrib["whenChoiceValue"],
                    }
            if conditionals:
                tag_info["conditionals"] = conditionals
            outputs[tag.attrib["name"]] = tag_info
        elif _is_input_tag(tag):
            inputs[tag.attrib["name"]] = {
                "type": tag.tag,
                "value": tag.attrib["value"].lstrip("$"),
            }
        if tag.tag not in _LABEL_TAGS:
            continue
        parent_name = _get_parent_output_tag_name(tag, outputs)
        if parent_name is not None:
            actual_value = tag.attrib.get("alias") or tag.attrib.get("value")
            if not actual_value:
                logger.debug(
                    'Inspecting tag {tag_name}... found no "value" or "alias" attributes.'.format(
                        tag_name=etree.tostring(tag, encoding="unicode").strip()[:50]
                    )
                )
            else:
                labels[parent_name][actual_value] = dict(tag.attrib)
    for output_tag, tag_info in outputs.items():
        tag_info["inputs"] = []
        for input_tag_name in tag_info["to_name"]:
            if input_tag_name not in inputs:
                logger.debug(
                    f"to_name={input_tag_name} is specified for output tag name={output_tag}, "
                    "but we can't find it among input tags"
                )
                continue
            tag_info["inputs"].append(inputs[input_tag_name])
        tag_info["labels"] = list(labels[output_tag])
        tag_info["labels_attrs"] = labels[output_tag]
    return outputs


def get_polygon_area(x, y):
    """https://en.wikipedia.org/wiki/Shoelace_formula"""

    assert len(x) == len(y)

    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_polygon_bounding_box(x, y):
    assert len(x) == len(y)

    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]
    

def get_cocomask_area(segmentation):
    return int(pycocotools.mask.area(segmentation))


def get_cocomask_bounding_box(segmentation):
    return pycocotools.mask.toBbox(segmentation).tolist()


def get_annotator(item, default=None, int_id=False):
    """Get annotator id or email from annotation"""
    annotator = item["completed_by"]
    if isinstance(annotator, dict):
        annotator = annotator.get("email", default)
        return annotator

    if isinstance(annotator, int) and int_id:
        return annotator

    return str(annotator)


def get_json_root_type(filename):
    char = "x"
    with open(filename, "r", encoding="utf-8") as f:
        # Read the file character by character
        while char != "":
            char = f.read(1)

            # Skip any whitespace
            if char.isspace():
                continue

            # If the first non-whitespace character is '{', it's a dict
            if char == "{":
                return "dict"

            # If the first non-whitespace character is '[', it's an array
            if char == "[":
                return "list"

            # If neither, the JSON file is invalid
            return "invalid"

    # If the file is empty, return "empty"
    return "empty"


def prettify_result(v):
    """
    :param v: list of regions or results
    :return: label name as is if there is only 1 item in result `v`, else list of label names
    """
    out = []
    tag_type = None
    for i in v:
        j = deepcopy(i)
        tag_type = j.pop("type")
        if tag_type == "Choices" and len(j["choices"]) == 1:
            out.append(j["choices"][0])
        elif tag_type == "TextArea" and len(j["text"]) == 1:
            out.append(j["text"][0])
        else:
            out.append(j)
    return out[0] if tag_type in ("Choices", "TextArea") and len(out) == 1 else out


def convert_annotation_to_yolo(label):
    """
    Convert LS annotation to Yolo format.

    Args:
        label (dict): Dictionary containing annotation information including:
            - width (float): Width of the object.
            - height (float): Height of the object.
            - x (float): X-coordinate of the top-left corner of the object.
            - y (float): Y-coordinate of the top-left corner of the object.

    Returns:
        tuple or None: If the conversion is successful, returns a tuple (x, y, w, h) representing
        the coordinates and dimensions of the object in Yolo format, where (x, y) are the center
        coordinates of the object, and (w, h) are the width and height of the object respectively.
    """

    if not ("x" in label and "y" in label and "width" in label and "height" in label):
        return None

    w = label["width"]
    h = label["height"]

    x = (label["x"] + w / 2) / 100
    y = (label["y"] + h / 2) / 100
    w = w / 100
    h = h / 100

    return x, y, w, h


def convert_annotation_to_yolo_obb(label, normalize=True):
    """
    Convert LS annotation to Yolo OBB format.

    Args:
        label (dict): Dictionary containing annotation information including:
            - original_width (int): Original width of the image.
            - original_height (int): Original height of the image.
            - x (float): X-coordinate of the top-left corner of the object in percentage of the original width.
            - y (float): Y-coordinate of the top-left corner of the object in percentage of the original height.
            - width (float): Width of the object in percentage of the original width.
            - height (float): Height of the object in percentage of the original height.
            - rotation (float, optional): Rotation angle of the object in degrees (default is 0).
        normalize (bool, optional): Whether to normalize the coordinates to the range [0, 1] (default is True).

    Returns:
        list of tuple or None: List of tuples containing the coordinates of the object in Yolo OBB format.
            Each tuple represents a corner of the bounding box in the order:
            (top-left, top-right, bottom-right, bottom-left).
    """

    if not (
        "original_width" in label
        and "original_height" in label
        and "x" in label
        and "y" in label
        and "width" in label
        and "height" in label
        and "rotation" in label
    ):
        return None

    org_width, org_height = label["original_width"], label["original_height"]
    x = label["x"] / 100 * org_width
    y = label["y"] / 100 * org_height
    w = label["width"] / 100 * org_width
    h = label["height"] / 100 * org_height

    rotation = math.radians(label.get("rotation", 0))
    cos, sin = math.cos(rotation), math.sin(rotation)

    coords = [
        (x, y),
        (x + w * cos, y + w * sin),
        (x + w * cos - h * sin, y + w * sin + h * cos),
        (x - h * sin, y + h * cos),
    ]

    # Normalize coordinates
    if normalize:
        return [(coord[0] / org_width, coord[1] / org_height) for coord in coords]
    else:
        return coords


def convert_yolo_obb_to_annotation(xyxyxyxy, original_width, original_height):
    """
    Convert YOLO Oriented Bounding Box (OBB) format to Label Studio format.

    Args:
        xyxyxyxy (list): List of 8 float values representing the absolute pixel coordinates
                         of the OBB in the format [x1, y1, x2, y2, x3, y3, x4, y4].
        original_width (int): Original width of the image.
        original_height (int): Original height of the image.

    Returns:
        dict: Dictionary containing the converted bounding box with the following keys:
              - x: X-coordinate of the top-left corner of the bounding box in percentage.
              - y: Y-coordinate of the top-left corner of the bounding box in percentage.
              - width: Width of the bounding box in percentage.
              - height: Height of the bounding box in percentage.
              - rotation: Rotation angle of the bounding box in degrees.
    """
    # Reshape the coordinates into a 4x2 matrix
    coords = np.array(xyxyxyxy, dtype=np.float64).reshape((4, 2))

    # Calculate the center of the bounding box
    center_x = np.mean(coords[:, 0])
    center_y = np.mean(coords[:, 1])

    # Calculate the width and height of the bounding box
    width = np.linalg.norm(coords[0] - coords[1])
    height = np.linalg.norm(coords[0] - coords[3])

    # Calculate the rotation angle
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]
    r = np.degrees(np.arctan2(dy, dx))

    # Find the top-left corner (x, y)
    top_left_x = (
        center_x
        - (width / 2) * np.cos(np.radians(r))
        + (height / 2) * np.sin(np.radians(r))
    )
    top_left_y = (
        center_y
        - (width / 2) * np.sin(np.radians(r))
        - (height / 2) * np.cos(np.radians(r))
    )

    # Normalize the values
    x = (top_left_x / original_width) * 100
    y = (top_left_y / original_height) * 100
    width = (width / original_width) * 100
    height = (height / original_height) * 100

    # Create the dictionary for Label Studio
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "rotation": r,
        "original_width": original_width,
        "original_height": original_height,
    }
