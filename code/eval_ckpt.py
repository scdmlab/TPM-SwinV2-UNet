"""Re-evaluate a saved checkpoint on the held-out test split with the trainer's own
data pipeline and metric code, so the result is directly comparable to result.json."""
import argparse, json, os, sys, importlib.util
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("T", os.path.join(HERE, "tpsm_train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data-root", default=r"D:/former_files/former_data (2)/former_data")
ap.add_argument("--code-root", default=r"D:/wildfire project/revision_2026/code_base/wildfire-TPSM-SwinV2-UNet-main")
ap.add_argument("--split", default=r"D:/wildfire project/revision_2026/results/split.json")
a = ap.parse_args()

cfg = T.variant_cfg(a.variant)
O = T.load_original(a.code_root)
split = json.load(open(a.split, encoding="utf-8"))
test_root = os.path.join(a.data_root, "Testing", "Testing")
ds_te = T.PatchDataset(test_root, split["test"], cfg["streams"], False, True, cfg["index_source"], None)
dl_te = DataLoader(ds_te, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
dev = torch.device("cuda")
m = T.SwinUNetV2Flex(O, streams=cfg["streams"], unet=cfg["unet"], ppm=cfg["ppm"], scse=cfg["scse"],
                     mmoe=cfg["mmoe"], experts=cfg["experts"], topk=cfg["topk"], gate_hint=cfg["gate_hint"],
                     num_classes=4, chans=ds_te.channels(), hint_mode=cfg["hint_mode"],
                     shared_gate=cfg["shared_gate"], hint_head=cfg["hint_head"]).to(dev)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
m.load_state_dict(ck["state_dict"])
m.set_gate_temperature(ck.get("gate_temperature", 0.5))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
te = T.evaluate(m, dl_te, dev, 4)
json.dump({"variant": a.variant, "ckpt": a.ckpt, "config": cfg, "test": te}, open(a.out, "w"), indent=1)
print("mIoU %.5f  OA %.5f  kappa %.5f  qwk %.5f  omae %.5f  iou %s" % (
    te["mIoU"], te["OA"], te["kappa"], te["qwk"], te["ordinal_mae"], [round(x, 5) for x in te["iou"]]))
