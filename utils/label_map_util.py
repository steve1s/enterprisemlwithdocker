# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Label map utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging

import numpy as np
from six import string_types
from six.moves import range
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

_LABEL_OFFSET = 1

def _validate_label_map(label_map):
    """Checks if a label map is valid.
    Args:
        label_map: StringIntLabelMap to validate.
    Raises:
        ValueError: if label map is invalid.
    """
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if (item.id == 0 and item.name != 'background' and item.display_name != 'background'):
            raise ValueError('Label map id 0 is reserved for the background label')

def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index

def get_max_label_map_index(label_map):
    return max([item.id for item in label_map.item])

def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({'id': class_id + label_id_offset, 'name': 'category_{}'.format(class_id + label_id_offset)})
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            category = {'id': item.id, 'name': name}
        # Handle missing optional fields gracefully
        if hasattr(item, 'HasField') and 'frequency' in item.DESCRIPTOR.fields_by_name:
            if item.HasField('frequency'):
                if hasattr(string_int_label_map_pb2, 'LVISFrequency'):
                    if item.frequency == string_int_label_map_pb2.LVISFrequency.Value('FREQUENT'):
                        category['frequency'] = 'f'
                    elif item.frequency == string_int_label_map_pb2.LVISFrequency.Value('COMMON'):
                        category['frequency'] = 'c'
                    elif item.frequency == string_int_label_map_pb2.LVISFrequency.Value('RARE'):
                        category['frequency'] = 'r'
        if hasattr(item, 'HasField') and 'instance_count' in item.DESCRIPTOR.fields_by_name:
            if item.HasField('instance_count'):
                category['instance_count'] = item.instance_count
            if item.keypoints:
                keypoints = {}
                list_of_keypoint_ids = []
                for kv in item.keypoints:
                    if kv.id in list_of_keypoint_ids:
                        raise ValueError('Duplicate keypoint ids are not allowed. Found {} more than once'.format(kv.id))
                    keypoints[kv.label] = kv.id
                    list_of_keypoint_ids.append(kv.id)
                category['keypoints'] = keypoints
            categories.append(category)
    return categories

def load_labelmap(path, validator=None):
    with tf.io.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    if validator is None:
        validator = _validate_label_map
    validator(label_map)
    return label_map

def get_label_map_dict(label_map_path_or_proto, use_display_name=False, fill_in_gaps_and_background=False, validator=None):
    if isinstance(label_map_path_or_proto, string_types):
        label_map = load_labelmap(label_map_path_or_proto)
    else:
        if validator is None:
            validator = _validate_label_map
        validator(label_map_path_or_proto)
        label_map = label_map_path_or_proto
    label_map_dict = {}
    for item in label_map.item:
        if use_display_name:
            label_map_dict[item.display_name] = item.id
        else:
            label_map_dict[item.name] = item.id
    if fill_in_gaps_and_background:
        values = set(label_map_dict.values())
        if 0 not in values:
            label_map_dict['background'] = 0
        if not all(isinstance(value, int) for value in values):
            raise ValueError('The values in label map must be integers in order to fill_in_gaps_and_background.')
        if not all(value >= 0 for value in values):
            raise ValueError('The values in the label map must be positive.')
        if len(values) != max(values) + 1:
            for value in range(1, max(values)):
                if value not in values:
                    label_map_dict[str(value)] = value
    return label_map_dict

def get_keypoint_label_map_dict(label_map_path_or_proto):
    if isinstance(label_map_path_or_proto, string_types):
        label_map = load_labelmap(label_map_path_or_proto)
    else:
        label_map = label_map_path_or_proto
    label_map_dict = {}
    for item in label_map.item:
        for kpts in item.keypoints:
            if kpts.label in label_map_dict.keys():
                raise ValueError('Duplicated keypoint label: %s' % kpts.label)
            if kpts.id in label_map_dict.values():
                raise ValueError('Duplicated keypoint ID: %d' % kpts.id)
            label_map_dict[kpts.label] = kpts.id
    return label_map_dict

def get_label_map_hierarchy_lut(label_map_path_or_proto, include_identity=False, validator=None):
    if isinstance(label_map_path_or_proto, string_types):
        label_map = load_labelmap(label_map_path_or_proto)
    else:
        if validator is None:
            validator = _validate_label_map
        validator(label_map_path_or_proto)
        label_map = label_map_path_or_proto
    hierarchy_dict = {'ancestors': collections.defaultdict(list), 'descendants': collections.defaultdict(list)}
    max_id = -1
    for item in label_map.item:
        max_id = max(max_id, item.id)
        for ancestor in item.ancestor_ids:
            hierarchy_dict['ancestors'][item.id].append(ancestor)
        for descendant in item.descendant_ids:
            hierarchy_dict['descendants'][item.id].append(descendant)
    def get_graph_relations_tensor(graph_relations):
        graph_relations_tensor = np.zeros([max_id, max_id])
        for id_val, ids_related in graph_relations.items():
            id_val = int(id_val) - _LABEL_OFFSET
            for id_related in ids_related:
                id_related -= _LABEL_OFFSET
                graph_relations_tensor[id_val, id_related] = 1
        if include_identity:
            graph_relations_tensor += np.eye(max_id)
        return graph_relations_tensor
    ancestors_lut = get_graph_relations_tensor(hierarchy_dict['ancestors'])
    descendants_lut = get_graph_relations_tensor(hierarchy_dict['descendants'])
    return ancestors_lut, descendants_lut

def create_categories_from_labelmap(label_map_path, use_display_name=True):
    label_map = load_labelmap(label_map_path)
    max_num_classes = max(item.id for item in label_map.item)
    return convert_label_map_to_categories(label_map, max_num_classes, use_display_name)

def create_category_index_from_labelmap(label_map_path, use_display_name=True):
    categories = create_categories_from_labelmap(label_map_path, use_display_name)
    return create_category_index(categories)

def create_class_agnostic_category_index():
    return {1: {'id': 1, 'name': 'object'}}
