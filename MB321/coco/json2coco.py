#!/usr/bin/env python
# json2coco.py
# chengyu wang, 2021527, XJTLU, last updated:

import argparse, collections, json, glob, datetime, uuid, os, sys, labelme, platform, PIL.Image
import os.path as osp
import numpy as np

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)

_UNAME_, is_debug, fp_debug = platform.uname(), False, ""
_IS_WINDOWS_, _RSTRIP_, _COPY_, _MOVE_, _DEL_ = False, '/', 'cp', 'mv', 'rm -rf'
if 'Windows' in _UNAME_[0]:
    _IS_WINDOWS_, _RSTRIP_, _COPY_, _MOVE_, _DEL_ = True, '\\', 'copy', 'move', 'del'
    print(f"!! windows will use '\\' in path which expect '/' in Ubuntu\n you will happy to use self defined 'join_path()' instead of 'osp.join()' \n\n")
TM = datetime.datetime.now().strftime("%m%d_%H%M%S")
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input_dir', default='raw', help='input annotated directory')
parser.add_argument('--output_dir', default=None, help='output dataset directory')
parser.add_argument('--labels', default='labels_chr.txt', help='labels file')  # , required=True)
args = parser.parse_args()
args.input_dir = args.input_dir.rstrip(_RSTRIP_)

def to_coco(input_dir, labels, _fp_out): # !!! ERROR !! _input='raw', _output='raw_coco', _labels='labels_chr.txt'
    if _fp_out is None:
        _fp_out = os.path.join(os.path.dirname(args.input_dir), f"{os.path.basename(args.input_dir)}_coco_{TM}")
    elif input_dir == _fp_out: input("same for input and output:\n{} \n press any key".format(_fp_out))
    print("json2coco: ~input={}, output={}, labels={}".format(args.input_dir, _fp_out, args.labels))
    jpg_dir = osp.join(_fp_out, 'JPEGImages')
    if osp.exists(_fp_out):
        r = input("Output directory already exists:{}\n input 'del' to delete, 'c' to continue, others to quit\n"
                  .format(_fp_out))
        if r in ['del', 'DEL', 'Del']:
            os.system("{} {}".format(_DEL_, _fp_out))
        elif r not in ['c', 'C']:
            raise ValueError("886")
        # ssys.exit(1)
    else:
        os.makedirs(_fp_out)
        os.makedirs(jpg_dir)
        print('Creating dataset:', _fp_out)

    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )
    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i # - 1  # starts with -1
        class_name = line.strip()
        if class_id == 0: # -1:
            assert class_name in ['__background__', '__ignore__'], "check 'labels_chr.txt', should be:\n \
                __background__\n chr" # '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    out_ann_file = osp.join(_fp_out, 'annotations.json')
    label_files = glob.glob(osp.join(input_dir, '*.json'))
    for image_id, label_file in enumerate(label_files):
        if image_id % 10 == 0 or image_id < 5 or image_id == len(label_files)-1:
            print('Generating dataset from [{}/{}]: {}'.format(image_id+1, len(label_files), label_file))
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = osp.join(
            _fp_out, 'JPEGImages', base + '.jpg'
        )

        img_file = osp.join(
            osp.dirname(label_file), label_data['imagePath']
        )
        img = np.asarray(PIL.Image.open(img_file).convert('RGB'))
        PIL.Image.fromarray(img).save(out_img_file)
        data['images'].append(dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=image_id,
        ))

        masks = {}                                     # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = shape.get('shape_type')
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data['annotations'].append(dict(
                id=len(data['annotations']),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[instance],
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)
# end to_coco

if __name__ == '__main__':
    to_coco(args.input_dir, args.labels, args.output_dir) # main()
