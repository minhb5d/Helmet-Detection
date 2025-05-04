import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset_Loader.voc_helmet_dataset import VOCDataset   
from Model.faster_rcnn import FasterRCNN        

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    filenames = [item[2] for item in batch]

    images = torch.stack(images, dim=0)
    targets = {
        'bboxes': [t['bboxes'] for t in targets],
        'labels': [t['labels'] for t in targets]
    }
    return images, targets, filenames

def main():
    # === Configurations === #
    dataset_root = "../helmet-detector"  #chỉnh nếu khác
    im_train_path = os.path.join(dataset_root, "train", "JPEGImages")
    ann_train_path = os.path.join(dataset_root, "train", "Annotations")

    train_params = {
        'seed': 1111,
        'num_epochs': 20,
        'lr_steps': [12, 16],
        'lr': 0.001,
        'ckpt_name': './checkpoints/faster_rcnn_helmet.pth'
    }

    model_params = {
        'backbone_out_channels': 512,
        'scales': [128, 256, 512],
        'aspect_ratios': [0.5, 1.0, 2.0],
        'rpn_bg_threshold': 0.3,
        'rpn_fg_threshold': 0.7,
        'rpn_nms_threshold': 0.7,
        'rpn_batch_size': 256,
        'rpn_pos_fraction': 0.5,
        'rpn_train_topk': 2000,
        'rpn_test_topk': 1000,
        'rpn_train_prenms_topk': 12000,
        'rpn_test_prenms_topk': 6000,
        'roi_batch_size': 128,
        'roi_pos_fraction': 0.25,
        'roi_iou_threshold': 0.5,
        'roi_low_bg_iou': 0.0, # increase it to 0.1 for hard negative
        'roi_nms_threshold': 0.5,
        'roi_topk_detections': 100,
        'roi_score_threshold': 0.05,
        'roi_pool_size': 7,
        'fc_inner_dim': 1024,
        'min_im_size': 600,
        'max_im_size': 1000
    }

    num_classes = 3  # background, helmet, no-helmet
    torch.manual_seed(train_params['seed'])

    # === Load dataset === #
    print("📦 Loading dataset...")
    train_dataset = VOCDataset('train', im_train_path, ann_train_path)
    print("✅ Number of training images:", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # === Setup model === #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FasterRCNN(model_params, num_classes).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=train_params['lr'], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params['lr_steps'], gamma=0.1)

    # === Training loop === #
    for epoch in range(train_params['num_epochs']):
        total_loss = 0.0

        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            targets = {k: [v_.to(device) for v_ in v] for k, v in targets.items()}

            optimizer.zero_grad()
            rpn_out, frcnn_out = model(images, targets)

            loss = (
                rpn_out['rpn_classification_loss'] +
                rpn_out['rpn_localization_loss'] +
                frcnn_out['frcnn_classification_loss'] +
                frcnn_out['frcnn_localization_loss']
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📌 Epoch {epoch+1}/{train_params['num_epochs']} - Loss: {total_loss:.4f}")
        lr_scheduler.step()

    # === Save model checkpoint === #
    os.makedirs(os.path.dirname(train_params['ckpt_name']), exist_ok=True)
    torch.save(model.state_dict(), train_params['ckpt_name'])
    print(f"✅ Model saved to: {train_params['ckpt_name']}")

if __name__ == '__main__':
    main()

