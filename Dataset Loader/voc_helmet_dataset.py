import os
import glob
import random
import torch
import torchvision
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        self.classes = ['background', 'helmet', 'no-helmet']
        self.label2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx2label = {idx: cls for idx, cls in enumerate(self.classes)}
        self.images_info = self.load_images_and_anns()

    def load_images_and_anns(self):
        im_infos = []
        for ann_file in tqdm(glob.glob(os.path.join(self.ann_dir, '*.xml'))):
            im_info = {}
            im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
            im_info['filename'] = os.path.join(self.im_dir, '{}.jpg'.format(im_info['img_id']))
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            im_info['width'] = width
            im_info['height'] = height
            detections = []
            for obj in root.findall('object'):
                name = obj.find('name').text.strip().lower()
                if name not in self.label2idx:
                    continue

                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text)) - 1
                ymin = int(float(bbox.find('ymin').text)) - 1
                xmax = int(float(bbox.find('xmax').text)) - 1
                ymax = int(float(bbox.find('ymax').text)) - 1

                #  Bỏ qua bbox không hợp lệ
                if xmax <= xmin or ymax <= ymin:
                    continue

                det = {
                    'label': self.label2idx[name],
                    'bbox': [xmin, ymin, xmax, ymax]
                }
                detections.append(det)

            im_info['detections'] = detections
            im_infos.append(im_info)
        return im_infos

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename']).convert("RGB")

        # Data augmentation: lật ảnh ngang
        to_flip = self.split == 'train' and random.random() < 0.5
        if to_flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        im_tensor = torchvision.transforms.ToTensor()(im)

        targets = {}
        bboxes = []
        labels = []

        for det in im_info['detections']:
            bbox = det['bbox']
            if to_flip:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                im_w = im_tensor.shape[-1]
                x1_new = im_w - x1 - w
                x2_new = x1_new + w
                bbox = [x1_new, y1, x2_new, y2]
            bboxes.append(bbox)
            labels.append(det['label'])

        targets['bboxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
        targets['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        return im_tensor, targets, im_info['filename']