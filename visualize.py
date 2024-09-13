import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image

from util.default_args import set_model_defaults, get_args_parser
from util.slconfig import SLConfig
import util.misc as utils
from models.dino import build_dino
import transforms as T

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_known_args()[0]
    set_model_defaults(args)

    device = torch.device(args.device)
    cfg = SLConfig.fromfile("./configs/DINO/DINO_4scale_convnext.py")
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
    if args.eval:
        args.semantic_aug = None
    model, criterion, postprocessors, scaler = build_dino(args)
    model.to(device)

    cp = torch.load(args.model_weight)
    msg = model.load_state_dict(cp, strict=False)
    print(msg)

    out_dir = "./vis"
    os.makedirs(out_dir, exist_ok=True)

    test_fp = args.input_image
    test_out = ".".join(test_fp.split(".")[:-1]) + "_predBox.png"

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    idx2name = {1: 'person', 2: 'rider', 3: 'car', 4: 'truck', 5: 'bus', 6: 'motorcycle', 7: 'bicycle'}
    colorbar = (np.arange(len(idx2name)) / (len(idx2name) - 1)) * 255
    colormap = cv2.applyColorMap(colorbar.astype(np.uint8), cv2.COLORMAP_RAINBOW)

    model.eval()
    with torch.no_grad():
        img = Image.open(test_fp).convert("RGB")
        sample, _ = transform(img, None)
        sample = sample.to(device)
        _, _, outputs = model([sample], None)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        sizes = torch.tensor([w, h, w, h])

        # plot predicted boxes
        class_probs = outputs["pred_logits"].softmax(dim=-1).cpu()
        pred_logits_val, pred_logits_ind = class_probs.max(dim=-1)
        valid_mask = pred_logits_val > 0.9
        pred_classes = pred_logits_ind[valid_mask]
        pred_boxes = outputs["pred_boxes"].cpu()[valid_mask] * sizes
        for box, label in zip(pred_boxes, pred_classes):
            cx, cy, w, h = box
            area = w * h
            x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
            img = cv2.rectangle(img, (x1.round().int().item(), y1.round().int().item()), (x2.round().int().item(), y2.round().int().item()), colormap[label.item() - 1][0].tolist(), 2)
        _ = cv2.imwrite(test_out, img)
