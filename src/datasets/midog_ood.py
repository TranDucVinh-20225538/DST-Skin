"""OpenMIBOOD MIDOG loaders for covariate-shift gap metrics.

ID = domain 1a test; OOD = cs-ID domains 1b + 1c (same labels, different scanners).
Yields (image, label, path) tuples so DST-Skin's feature extractor does not need
OpenOOD dict batches. OpenOODDictAdapter wraps dict-style samples if needed.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

MIDOG_NUM_CLASSES = 3
DATA_ROOT = "data/raw/midog"
IMGLIST_DIR = "data/benchmark_imglist/midog"
CKPT_PATH = "data/models/midog/midog_classifier.pth"
CKPT_URL = (
    "https://zenodo.org/records/14982267/files/midog_classifier.pth?download=1"
)
FIGSHARE_MAP = Path(__file__).with_name("midog_figshare_map.json")
JSON_FIGSHARE_ID = "41265615"

# OpenMIBOOD domain index ranges into MIDOGpp.json["images"]
DOMAIN_SPLIT = (
    ("1a", 0, 50, ""),
    ("1b", 50, 100, "csid"),
    ("1c", 100, 150, "csid"),
)

MIDOG_MEAN = [0.712, 0.496, 0.756]
MIDOG_STD = [0.167, 0.167, 0.110]

IMGLIST_FILES = {
    "train": "train_midog.txt",
    "id_val": "valid_midog.txt",
    "id_test": "test_midog.txt",
    "csid_1b": "test_midog_1b.txt",
    "csid_1c": "test_midog_1c.txt",
}


class OpenOODDictAdapter(Dataset):
    """Convert OpenOOD ImglistDataset dict samples to (image, label, path)."""

    def __init__(self, inner: Dataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        sample = self.inner[idx]
        if isinstance(sample, dict):
            image = sample.get("data", sample.get("image"))
            label = int(sample["label"])
            path = str(sample.get("image_name", sample.get("path", idx)))
            return image, label, path
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            path = sample[2] if len(sample) > 2 else str(idx)
            return sample[0], int(sample[1]), path
        raise TypeError(f"Unsupported sample type: {type(sample)}")


class ImglistTupleDataset(Dataset):
    """OpenOOD imglist format (`relpath label`) → DST-Skin (image, label, path)."""

    def __init__(self, imglist_path: str | Path, data_root: str | Path, transform):
        self.data_root = Path(data_root)
        self.transform = transform
        self.records: list[tuple[str, int]] = []
        with open(imglist_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, label_s = line.split(" ", 1)
                self.records.append((rel, int(label_s)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rel, label = self.records[idx]
        path = self.data_root / rel
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, rel


def _input_size(backbone: str | None, input_size: int | None = None) -> int:
    """Public OpenMIBOOD R50 is native 50×50 (backbone=None).

    In-house zoo: ResNets 224, EffB3 300, other CNNs 224. Official EffV2-S is
    224 (native 384 is the Camelyon/skin resolution trap) so Kendall W can
    include the tagged `efficientnet_v2_s_224` stem.
    """
    if input_size is not None:
        return int(input_size)
    if backbone is None:
        return 50
    if backbone == "efficientnet_v2_s":
        return 224
    from src.models.cnn_family import input_size as cnn_input

    return cnn_input(backbone)


def build_transform(
    backbone: str | None = None,
    train: bool = False,
    input_size: int | None = None,
) -> transforms.Compose:
    """MIDOG mean/std (adapter). Native 50×50 crops; upsample for in-house zoo."""
    size = _input_size(backbone, input_size=input_size)
    normalize = transforms.Normalize(MIDOG_MEAN, MIDOG_STD)
    if size == 50:
        spatial = [transforms.Resize(50)]
        if train:
            spatial.append(transforms.RandomHorizontalFlip())
        else:
            spatial.append(transforms.CenterCrop(50))
        return transforms.Compose(spatial + [transforms.ToTensor(), normalize])
    if train:
        resize = 320 if size == 300 else 256
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    resize = size if size == 300 else 256
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])


def imglist_path(split: str) -> Path:
    return Path(IMGLIST_DIR) / IMGLIST_FILES[split]


def data_ready(root: str = DATA_ROOT) -> Path:
    return Path(root) / "RELEASE_id_csid.txt"


def split_summary() -> dict[str, int | str]:
    summary: dict[str, int | str] = {
        "id_eval": "test_1a",
        "ood": "csid_1b+1c",
        "num_classes": MIDOG_NUM_CLASSES,
    }
    for split, fname in IMGLIST_FILES.items():
        path = Path(IMGLIST_DIR) / fname
        n = sum(1 for line in open(path) if line.strip()) if path.exists() else -1
        summary[split] = n
    summary["ood_n"] = int(summary["csid_1b"]) + int(summary["csid_1c"])
    return summary


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    include_ood: bool = True,
    data_root: str = DATA_ROOT,
    backbone: str | None = None,
    transform=None,
    input_size: int | None = None,
) -> dict[str, DataLoader]:
    """train / id_val / ID-test (1a) / optional cs-ID (1b+1c). Same imglists as OpenMIBOOD.

    `transform`, if given, overrides MIDOG mean/std and native 50×50 — used for
    frozen WSI-FMs that expect ImageNet-normalized 224² tiles.
    """
    if transform is not None:
        eval_tf = train_tf = transform
    else:
        eval_tf = build_transform(backbone, train=False, input_size=input_size)
        train_tf = (
            build_transform(backbone, train=True, input_size=input_size)
            if backbone
            else eval_tf
        )
    loaders: dict[str, DataLoader] = {
        "train": _make_loader(
            ImglistTupleDataset(imglist_path("train"), data_root, train_tf),
            batch_size,
            num_workers,
            shuffle=True,
        ),
        "id_val": _make_loader(
            ImglistTupleDataset(imglist_path("id_val"), data_root, eval_tf),
            batch_size,
            num_workers,
            shuffle=False,
        ),
        "id_test": _make_loader(
            ImglistTupleDataset(imglist_path("id_test"), data_root, eval_tf),
            batch_size,
            num_workers,
            shuffle=False,
        ),
    }
    if include_ood:
        ood = ConcatDataset([
            ImglistTupleDataset(imglist_path("csid_1b"), data_root, eval_tf),
            ImglistTupleDataset(imglist_path("csid_1c"), data_root, eval_tf),
        ])
        loaders["ood"] = _make_loader(ood, batch_size, num_workers, shuffle=False)
    return loaders


def _figshare_map() -> dict[str, str]:
    return json.loads(Path(FIGSHARE_MAP).read_text())


def _filename_to_id(mapping: dict[str, str]) -> dict[str, str]:
    return {fname: fid for fid, fname in mapping.items()}


def _download_file(url: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return
        except Exception as exc:
            last_err = exc
            print(f"  retry {attempt}/{retries} {dest.name}: {exc}", flush=True)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Failed to download {url}") from last_err


def download_checkpoint(dest: str | Path = CKPT_PATH) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 90_000_000:
        print(f"Checkpoint already present: {dest} ({dest.stat().st_size} bytes)")
        return dest
    print(f"Downloading MIDOG ResNet-50 checkpoint → {dest}", flush=True)
    _download_file(CKPT_URL, dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)", flush=True)
    return dest


def validate_patch(img_dims, annotations, additional_bbox) -> bool:
    if (
        additional_bbox[0] < 0
        or additional_bbox[1] < 0
        or additional_bbox[2] >= img_dims[0]
        or additional_bbox[3] >= img_dims[1]
    ):
        return False
    for annotation in annotations:
        bbox = annotation["bbox"]
        if bbox[0] < additional_bbox[0] < bbox[2] and bbox[1] < additional_bbox[1] < bbox[3]:
            return False
        if bbox[0] < additional_bbox[2] < bbox[2] and bbox[1] < additional_bbox[3] < bbox[3]:
            return False
    return True


def _needed_filenames(midog_json: dict) -> list[str]:
    names = []
    for _domain, start, end, _sub in DOMAIN_SPLIT:
        for idx in range(start, end):
            names.append(midog_json["images"][idx]["file_name"])
    return names


def download_id_csid(
    data_root: str = DATA_ROOT,
    keep_wsi: bool = False,
) -> Path:
    """Download only domains 1a/1b/1c (~150 WSIs), crop 50×50 patches, write marker."""
    root = Path(data_root)
    wsi_dir = root / "_wsi"
    root.mkdir(parents=True, exist_ok=True)
    wsi_dir.mkdir(parents=True, exist_ok=True)

    mapping = _figshare_map()
    name_to_id = _filename_to_id(mapping)
    json_path = wsi_dir / "MIDOGpp.json"
    if not json_path.exists():
        print("Downloading MIDOGpp.json ...", flush=True)
        _download_file(
            f"https://ndownloader.figshare.com/files/{JSON_FIGSHARE_ID}",
            json_path,
        )
    midog = json.loads(json_path.read_text())
    needed = _needed_filenames(midog)
    print(f"MIDOG ID+cs-ID: {len(needed)} WSIs (not full 503 / 65GB)", flush=True)

    pending = []
    for fname in needed:
        dest = wsi_dir / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            continue
        pending.append((fname, name_to_id[fname], dest))
    print(f"WSIs already present: {len(needed) - len(pending)}; to download: {len(pending)}", flush=True)

    def _one(item):
        fname, fid, dest = item
        _download_file(f"https://ndownloader.figshare.com/files/{fid}", dest)
        return fname, dest.stat().st_size

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_one, item): item[0] for item in pending}
        for fut in as_completed(futures):
            fname = futures[fut]
            done += 1
            _name, size = fut.result()
            print(f"[{done}/{len(pending)}] {fname} ({size / 1e6:.1f} MB)", flush=True)

    for domain_name, start, end, subfolder in DOMAIN_SPLIT:
        print(f"Cropping domain {domain_name} ...", flush=True)
        out_dir = root / subfolder / domain_name if subfolder else root / domain_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_index in range(start, end):
            img_data = midog["images"][img_index]
            img_name = img_data["file_name"]
            img_id = img_data["id"]
            img_dir = out_dir / f"{img_id:03d}"
            img_dir.mkdir(parents=True, exist_ok=True)
            image = Image.open(wsi_dir / img_name)
            img_dims = (img_data["width"], img_data["height"])
            annotations = [
                ann for ann in midog["annotations"] if ann["image_id"] == img_id
            ]
            for annotation in annotations:
                label = annotation["category_id"]
                bbox = annotation["bbox"]
                ann_id = annotation["id"]
                if bbox[2] - bbox[0] != 50 or bbox[3] - bbox[1] != 50:
                    continue
                crop_path = img_dir / f"{img_id:03d}_{ann_id}_{label}.tiff"
                if not crop_path.exists():
                    image.crop(bbox).save(crop_path)
                additional_bbox = [bbox[0] + 100, bbox[1], bbox[2] + 100, bbox[3]]
                if validate_patch(img_dims, annotations, additional_bbox):
                    bg_path = img_dir / f"{img_id:03d}_{ann_id}_0.tiff"
                    if not bg_path.exists():
                        image.crop(tuple(additional_bbox)).save(bg_path)
            image.close()

    missing = []
    for split in ("train", "id_test", "csid_1b", "csid_1c"):
        with open(imglist_path(split)) as f:
            for line in f:
                rel = line.strip().split(" ", 1)[0]
                if not (root / rel).exists():
                    missing.append(rel)
    if missing:
        preview = "\n  ".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} cropped files missing vs imglists, e.g.\n  {preview}"
        )

    if not keep_wsi:
        print("Removing raw WSIs to save ~20GB ...", flush=True)
        for fname in needed:
            path = wsi_dir / fname
            if path.exists():
                path.unlink()

    marker = data_ready(data_root)
    marker.write_text(
        "MIDOG domains 1a/1b/1c cropped for OpenMIBOOD imglists\n"
        + json.dumps(split_summary(), indent=2)
        + "\n"
    )
    print(f"MIDOG ID+cs-ID ready: {marker}", flush=True)
    return marker
