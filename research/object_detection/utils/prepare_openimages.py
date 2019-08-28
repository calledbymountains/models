import os
import glob
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--train_images_main_dir', required=True, type=str,
                    help='Path where the training images for OpenImages are stored.')
parser.add_argument('--val_images_dir', required=True, type=str,
                    help='Path where the validation images for OpenImages are stored.')
parser.add_argument('--test_images_dir', required=True, type=str,
                    help='Path where the test images for OpenImages are stored.')
parser.add_argument('--train_bbox', required=True, type=str,
                    help='File storing the training bounding box annotations for OpenImages.')
parser.add_argument('--val_bbox', required=True, type=str,
                    help='File storing the validation bounding box annotations for OpenImages.')
parser.add_argument('--test_bbox', required=True, type=str,
                    help='File storing the testing bounding box annotations for OpenImages.')
parser.add_argument('--classname_file', required=True, type=str,
                    help='File storing the class names and class labels for OpenImages.')
parser.add_argument('--only_person', dest='only_person', action='store_true',
                    help='When used, only person class is considered (Set by default)')
parser.add_argument('--all_classes', dest='only_person', action='store_false',
                    help='When used all object classes are considered (NOT SET by default)')
parser.set_defaults(only_person=True)
parser.add_argument('--output_dir', required=True, type=str,
                    help='Folder to store the results.')
parser.add_argument('--use_slurm_batch', dest='use_slurm_batch', action='store_true',
                    help='Use this only when you are submitting the job on a slurm scheduler. (NOT SET by default).')
parser.add_argument('--no_use_slurm', dest='use_slurm_batch', action='store_false',
                    help='Use this only when you are not submitting the job on a slurm scheduler. (SET by default).')
parser.set_defaults(use_slurm_batch=False)
parser.add_argument('--slurm_index', required=False, type=int,
                    help='Index of the slurm batch job. Must be provided when --use_slurm_batch is set.'
                         'The value must be 0, 1 or 2.')

PERSONCLASSES = ['Man', 'Person', 'Woman', 'Boy', 'Girl']


def train_image_subfolders(train_image_folder):
    subfolders = ['train_0{}'.format(x) for x in range(9)]
    subfolders = list(map(lambda x:
                          os.path.join(train_image_folder, x), subfolders))
    return subfolders


def merge_two_dataframes(frame1, frame2):
    df = pd.concat([frame1, frame2], axis=1)
    return df


def check_existence(name, objtype='file'):
    if objtype == 'file':
        assert (os.path.isfile(name)), "The file {} does not exist.".format(name)
    else:
        assert (os.path.isdir(name)), "The folder {} does not exist.".format(name)
    return None


def get_imagedetails(folders):
    imagedict = dict()
    for folder in folders:
        files = glob.glob(
            os.path.join(folder, '**', '*.jpg'),
            recursive=True
        )
        for f in files:
            imageid = os.path.splitext(
                os.path.basename(f)
            )[0]
            imagedict[imageid] = f

    return imagedict


if __name__ == "__main__":
    args = parser.parse_args()
    train_images_main_dir = args.train_images_main_dir
    val_images_dir = args.val_images_dir
    test_images_dir = args.test_images_dir
    train_bbox = args.train_bbox
    val_bbox = args.val_bbox
    test_bbox = args.test_bbox
    classname_file = args.classname_file
    only_person = args.only_person
    output_dir = args.output_dir
    use_slurm_batch = args.use_slurm_batch
    print(args.slurm_index)
    if use_slurm_batch and  args.slurm_index is None:
        parser.error('When --use_slurm_batch is on, one must provide slurm_index.')
    else:
        slurm_index = args.slurm_index
    train_image_folders = train_image_subfolders(train_images_main_dir)
    [check_existence(x, 'folder') for x in train_image_folders]
    check_existence(val_images_dir, 'folder')
    check_existence(test_images_dir, 'folder')
    check_existence(output_dir, 'folder')
    check_existence(train_bbox, 'file')
    check_existence(val_bbox, 'file')
    check_existence(test_bbox, 'file')
    investigation_list = [
        (train_image_folders, train_bbox, 'training'),
        ([val_images_dir], val_bbox, 'validation'),
        ([test_images_dir], test_bbox, 'testing')
    ]
    logging.info('Reading the class name and label details...')
    classname_df = pd.read_csv(classname_file, names=['encoded_name', 'classname'])
    if only_person:
        logging.info('Only persons were requested as an object class.')
        logging.info('Selecting rows which correspond to persons.')
        logging.info('In OpenImages this corresponds to following classes :')
        logging.info('{}'.format(",".join(PERSONCLASSES)))
        classname_df = classname_df[classname_df.classname.isin(PERSONCLASSES)]

    if use_slurm_batch:
        investigation_list = [investigation_list[slurm_index]]
    for folders, bbox_file, subset in investigation_list:
        logging.info('Processing the {} subset.'.format(subset))
        bbox_df = pd.read_csv(bbox_file)
        bbox_df = bbox_df[bbox_df.LabelName.isin(classname_df.encoded_name.tolist())]
        bbox_df = bbox_df.groupby(by='ImageID').agg({
            'XMin': lambda x: list(x),
            'XMax': lambda x: list(x),
            'YMin': lambda x: list(x),
            'YMax': lambda x: list(x),
            'LabelName': lambda x: list(x)
        })
        bbox_df = bbox_df.reset_index()
        classnames = list()
        for rownum in range(bbox_df.shape[0]):
            encoded_names = bbox_df['LabelName'][rownum]
            if only_person:
                actual_names = ['Person'] * len(encoded_names)
                bbox_df['LabelName'][rownum] = classname_df[classname_df.classname == 'Person']['encoded_name'].tolist() * len(
                    encoded_names)
            else:
                actual_names = [
                    classname_df[classname_df.encoded_name == x]['classname'].tolist()[0]
                    for x in encoded_names
                ]
            classnames.append(actual_names)

        bbox_df['classnames'] = classnames

        imagedict = get_imagedetails(folders)
        bbox_df['ImageID'] = bbox_df['ImageID'].apply(lambda x: imagedict[x])
        savefilename = os.path.join(output_dir, '{}_list.csv'.format(subset))
        bbox_df.to_csv(savefilename, index=False)
