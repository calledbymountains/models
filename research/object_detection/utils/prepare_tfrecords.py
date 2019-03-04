import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import glob
import itertools
import cv2
from bidict import bidict
import tensorflow as tf
import multiprocessing as mp
from utils import labelmap_util

"""
The below code will create TFRecords for the SSM codebase.
You will need to create an imagelist file.
Currently we only allow one imagelist file for one dataset.
"""

parser = argparse.ArgumentParser(
    prog='Code to prepare TFRecords for SSM system.')
parser.add_argument('-A', '--annlist', required=True,
                    help='Fullpath to the imagelist file.')
parser.add_argument('-M', '--labelmap', required=True,
                    help='Full path to the labelmap file.')
parser.add_argument('-S', '--numshards', required=False, default=100,
                    help='Number of shards to split TFRecords into.')
parser.add_argument('-N', '--basename', required=True,
                    help='Base name of the TFRecords. If basename is COCO2017 and number of shards'
                         'is 3, then the created TFRecords will be named as '
                         'COCO2017-000001-of-000003 and so on.')
parser.add_argument('-P', '--savepath', required=True,
                    help='Path where the TFRecords should be stored. Will be created if one '
                         'does not already exist.')
parser.add_argument('-BB', '--boundingboxes', required=False, action='store_true', default=False,
                    help='If specified boundingboxes will be encoded. Only use this option'
                         'if you want to create TFRecords during testing phase.')
parser.add_argument('-NB', '--normalizeboxes', required=False, action='store_true',
                    default=False,
                    help='If specified, bounding boxes will be normalized between 0 and 1. Only specify'
                         'this option if your bounding boxes are not normalized already'
                         'such as in MSCOCO. Specifying this for a dataset like '
                         'OpenImages will result in incorrect training/evaluation as '
                         'bounding boxes are already normalized for OpenImages')
parser.add_argument('-D', '--use_display_name', required=False, action='store_true',
                    default=False,
                    help='This option is especially put for OpenImages. Do not use it for other datasets.')

NORMALIZE_BOXES = None

LABEL_MAP = None

ENCODE_BOXES = None


def parse_annotation_file(filename, boundingboxes):
    """
    Parses an annotation file for creating the TFRecords.
    Each line in the annotation file must have the following format --

    <FULL_PATH_TO_IMAGE_FILE><TAB CHAR><xmin,ymin,xmax,ymax,labelint><TAB CHAR>.....\n
    :param filename: Full path of the annotation file.
    :param labelmap: A bidict corresponding to the mapping between label names and label integers.
    :return: A dictionary mapping image file names to their annotations. For a specific image file F, the returned dictionary D will behave as follows :
            D[F] = {'labelname1' : [[x11,y11,x12,y12],[x21,y21,x22,y22]...],
                    and so on }

    """
    parsed_dict = dict()
    counter = 0
    if not boundingboxes:
        tf.logging.info('No bounding boxes will be parsed')
    else:
        tf.logging.info('Bounding boxes will be parsed. If bounding boxes are not found, there will be an error.')
    for line in open(filename, 'r'):
        counter += 1
        line = line.strip()
        line = line.split('\t')
        parsed_dict[line[0]] = dict()
        if boundingboxes:
            parsed_dict[line[0]]['maskfile'] = line[1]
            for bbox_info in line[2:]:
                info = bbox_info.split(',')
                if not len(info) == 5:
                    raise ValueError(
                        'The annotation file seems incorrectly formatted at line {} in {}'.format(
                            counter, filename))
                xmin, ymin, xmax, ymax, label = list(map(float, info))

                if not xmin < xmax:
                    raise ValueError('xmin > xmax at line {}. xmin = {} and xmax = {}'.format(counter, xmin, xmax))

                if not ymin < ymax:
                    raise ValueError('ymin > ymax at line {}. ymin = {} and ymax = {}'.format(counter, ymin, ymax))
                label = int(label)
                try:
                    label_name = LABEL_MAP.inv[label]
                except:
                    raise ValueError(
                        'The label number {} was not found in the labelmap.'.format(
                            label))

                if not label_name in parsed_dict[line[0]].keys():
                    parsed_dict[line[0]][label_name] = [[xmin, ymin, xmax, ymax]]
                else:
                    parsed_dict[line[0]][label_name].append(
                        [xmin, ymin, xmax, ymax])
        elif len(line) == 1:
            raise ValueError('In line {} of {}, there were no bounding boxes.'.format(counter, filename))

    return parsed_dict


def get_ann_files(annfiles_glob):
    """
    Given a glob pattern of annotation files, returs a list of all files matching the glob.
    :param annfiles_glob: Glob pattern of annotation files.
    :return: A list containing the
    """
    annfiles = glob.glob(annfiles_glob)
    if annfiles is None:
        raise ValueError('No annotation files were found.')

    return annfiles


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def prepare_image(image_path, annotation):
    """
    Reads an image and converts it to bytestring. Also finds its height and width
    and returns a dict containing the bytestring, height and width.
    :param image_path: Full path to the image
    :return: A dictionary with image : image bytestring, height : image height and width : image width
    """
    if not os.path.exists(image_path):
        raise ValueError(
            'The image {} does not exist. Please check.'.format(image_path))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ext = os.path.splitext(os.path.basename(image_path))[1]
    height = image.shape[0]
    width = image.shape[1]

    image = cv2.imencode(ext, image)[1].tostring()
    mask_image = None
    if 'maskfile' in annotation.keys():
        mask_file = annotation['maskfile']
        mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask_ext = os.path.splitext(os.path.basename(mask_file))[1]
        mask_image = cv2.imencode(mask_ext, mask_image)[1].tostring()
    return dict(image=image, height=height, width=width,
                filename=os.path.basename(image_path),
                mask_image=mask_image)


def create_example(image_dict, annotation_dict):
    image_height = image_dict['height']
    image_width = image_dict['width']
    filename = image_dict['filename']
    encoded_image = image_dict['image']
    mask_image = image_dict['mask_image']
    if mask_image is None:
        mask_image = ''.encode('utf-8')
    object_names = list(annotation_dict.keys())
    object_names.remove('maskfile')
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    category_names = []
    category_labels = []
    if ENCODE_BOXES:
        for obj_name in object_names:
            min_x = [x[0] for x in annotation_dict[obj_name]]
            max_x = [x[2] for x in annotation_dict[obj_name]]
            min_y = [x[1] for x in annotation_dict[obj_name]]
            max_y = [x[3] for x in annotation_dict[obj_name]]
            xmin += min_x
            ymin += min_y
            xmax += max_x
            ymax += max_y
            category_names += [obj_name] * len(min_x)
            category_labels += [LABEL_MAP[obj_name]] * len(min_x)

        if NORMALIZE_BOXES:
            xmin = [x / (float(image_width)-1.0) for x in xmin]
            xmax = [x / (float(image_width) - 1.0) for x in xmax]
            ymin = [x / (float(image_height) - 1.0) for x in ymin]
            ymax = [x / (float(image_height) - 1.0) for x in ymax]
        category_names = list(map(lambda x: str.encode(x), category_names))
        category_labels = list(map(int, category_labels))

    diff_x = [a - b for a, b in list(zip(xmax, xmin))]
    diff_y = [a - b for a, b in list(zip(ymax, ymin))]
    if not all([x >= 0 for x in diff_x]) or not all([x >= 0 for x in diff_y]):
        raise ValueError('There are annotation errors.')
    feature_dict = {
        'image/height':
            int64_feature(image_height),
        'image/width':
            int64_feature(image_width),
        'image/filename':
            bytes_feature(filename.encode('utf8')),
        'image/encoded':
            bytes_feature(encoded_image),
        'image/object/bbox/xmin':
            float_list_feature(xmin),
        'image/object/bbox/xmax':
            float_list_feature(xmax),
        'image/object/bbox/ymin':
            float_list_feature(ymin),
        'image/object/bbox/ymax':
            float_list_feature(ymax),
        'image/object/class/label': int64_list_feature(category_labels),
        'image/object/class/text':
            bytes_list_feature(category_names),
        'image/object/mask' : bytes_feature(mask_image)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_single_record(annotation_dict, tfrecord_name):
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    for image_path, annotation in annotation_dict.items():
        image_dict = prepare_image(image_path, annotation)
        example = create_example(image_dict, annotation)
        writer.write(example.SerializeToString())
    writer.close()
    return None


def chunks_of_dict(dict_input, num_chunks):
    it = iter(dict_input)
    step_size = int(len(dict_input) / num_chunks)
    for i in range(0, len(dict_input), step_size):
        yield {k: dict_input[k] for k in itertools.islice(it, step_size)}


def create_tfrecords(annfiles, numshards, basename, savepath):
    annotation_dict = dict()
    for ann_file in annfiles:
        annotation_dict.update(parse_annotation_file(ann_file, ENCODE_BOXES))

    annotation_chunks = []
    for chunk in chunks_of_dict(annotation_dict, numshards):
        annotation_chunks.append(chunk)

    os.makedirs(savepath, exist_ok=True)
    tfrecord_names = []
    for ind, _ in enumerate(range(numshards)):
        name = '{}-{}-of-{}.tfrecord'.format(basename, str(ind + 1).zfill(6),
                                             str(numshards).zfill(6))
        name = os.path.join(savepath, name)
        tfrecord_names.append(name)

    pool = mp.Pool(processes=mp.cpu_count())
    args_to_pool = list(zip(annotation_chunks, tfrecord_names))
    pool.starmap(create_single_record, args_to_pool)
    pool.close()
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    annotation_files_glob = args.annlist
    labelmap = args.labelmap
    numshards = int(args.numshards)
    basename = args.basename
    savepath = args.savepath
    ENCODE_BOXES = bool(args.boundingboxes)
    normalizeboxes = bool(args.normalizeboxes)
    use_display_name = bool(args.use_display_name)

    annfiles = get_ann_files(annotation_files_glob)
    if numshards <= 0:
        raise ValueError('Number of shards must be a positive integer.')

    NORMALIZE_BOXES = normalizeboxes

    LABEL_MAP = labelmap_util.get_label_map_dict(labelmap, use_display_name)
    LABEL_MAP = bidict(LABEL_MAP)
    create_tfrecords(annfiles, numshards, basename, savepath)
