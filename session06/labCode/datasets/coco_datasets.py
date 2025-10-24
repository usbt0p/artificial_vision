"""
COCO dataset loader for object detection
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
import urllib.request
import zipfile
from tqdm import tqdm
import os


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class COCODetectionDataset(Dataset):
    """
    COCO Dataset for object detection
    
    Returns images with bounding boxes and labels
    """
    
    @staticmethod
    def _download_url(url, output_path):
        """Download file with progress bar"""
        print(f"Downloading {url.split('/')[-1]}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    @staticmethod
    def _extract_zip(zip_path, extract_to):
        """Extract zip file"""
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction complete!")
    
    @staticmethod
    def _download_coco_split(data_root, split='val'):
        """
        Download COCO dataset split
        
        Args:
            data_root: Root directory for COCO data (e.g., 'data/coco')
            split: 'train' or 'val'
        """
        data_root = Path(data_root)
        data_root.mkdir(parents=True, exist_ok=True)
        
        base_url = "http://images.cocodataset.org"
        
        # Determine what to download
        images_url = f"{base_url}/zips/{split}2017.zip"
        annotations_url = f"{base_url}/annotations/annotations_trainval2017.zip"
        
        images_zip = data_root / f"{split}2017.zip"
        annotations_zip = data_root / "annotations_trainval2017.zip"
        images_dir = data_root / f"{split}2017"
        annotations_dir = data_root / "annotations"
        
        print("="*60)
        print(f"COCO Dataset Auto-Download ({split} split)")
        print("="*60)
        
        # Download images if needed
        if not images_dir.exists():
            if not images_zip.exists():
                size = "~18GB" if split == 'train' else "~1GB"
                print(f"\n{split}2017 images not found. Downloading ({size})...")
                try:
                    COCODetectionDataset._download_url(images_url, images_zip)
                except Exception as e:
                    print(f"Error downloading images: {e}")
                    print(f"Please download manually from: {images_url}")
                    raise
            
            # Extract images
            COCODetectionDataset._extract_zip(images_zip, data_root)
            print(f"Removing {images_zip.name} to save space...")
            images_zip.unlink()
        else:
            print(f"\n{split}2017 images already exist")
        
        # Download annotations if needed
        if not annotations_dir.exists():
            if not annotations_zip.exists():
                print(f"\nAnnotations not found. Downloading (~250MB)...")
                try:
                    COCODetectionDataset._download_url(annotations_url, annotations_zip)
                except Exception as e:
                    print(f"Error downloading annotations: {e}")
                    print(f"Please download manually from: {annotations_url}")
                    raise
            
            # Extract annotations
            COCODetectionDataset._extract_zip(annotations_zip, data_root)
            print(f"Removing {annotations_zip.name} to save space...")
            annotations_zip.unlink()
        else:
            print(f"Annotations already exist")
        
        print("="*60)
        print("Dataset ready!")
        print("="*60)
    
    def __init__(self, img_dir, ann_file, transform=None, is_train=True, filter_empty=True):
        """
        Args:
            img_dir: Path to images directory
            ann_file: Path to COCO annotation JSON file
            transform: Data augmentation transforms
            is_train: Whether this is training set
            filter_empty: Whether to filter images without annotations
        """
        self.img_dir = Path(img_dir)
        ann_file_path = Path(ann_file)
        
        # Auto-download if dataset doesn't exist
        if not self.img_dir.exists() or not ann_file_path.exists():
            print(f"\nCOCO dataset not found at {self.img_dir} or {ann_file_path}")
            
            # Determine split from path
            if 'train' in str(self.img_dir):
                split = 'train'
            elif 'val' in str(self.img_dir):
                split = 'val'
            else:
                raise ValueError(f"Cannot determine split from path: {self.img_dir}")
            
            # Get data root (parent of train2017/val2017)
            data_root = self.img_dir.parent
            
            print(f"Attempting to download COCO {split} split...")
            print("This may take a while depending on your internet connection.")
            
            response = input(f"\nDownload COCO {split} dataset? [y/N]: ")
            if response.lower() != 'y':
                raise FileNotFoundError(
                    f"COCO dataset not found and download cancelled.\n"
                    f"Please download manually:\n"
                    f"  Images: http://images.cocodataset.org/zips/{split}2017.zip\n"
                    f"  Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n"
                    f"Extract to: {data_root}"
                )
            
            # Download the dataset
            self._download_coco_split(data_root, split)
        
        self.coco = COCO(str(ann_file_path))
        self.transform = transform
        self.is_train = is_train
        
        # Get all image IDs
        self.img_ids = list(self.coco.imgs.keys())
        
        # Filter images without annotations (optional)
        if filter_empty and is_train:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
            ]
        
        # Create contiguous category ID mapping (COCO IDs are not contiguous)
        self.coco_ids = sorted(self.coco.getCatIds())
        self.coco_id_to_class = {coco_id: idx for idx, coco_id in enumerate(self.coco_ids)}
        self.class_to_coco_id = {idx: coco_id for coco_id, idx in self.coco_id_to_class.items()}
        
        print(f"Loaded COCO dataset: {len(self.img_ids)} images, {len(self.coco_ids)} classes")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Get one sample from dataset
        
        Returns:
            image: Tensor [3, H, W]
            boxes: Tensor [N, 4] in XYXY format, normalized to [0, 1]
            labels: Tensor [N] with class indices [0, num_classes-1]
            image_id: int
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        img_w, img_h = image.size
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            # Skip crowd annotations
            if ann.get('iscrowd', 0) == 1:
                continue
            
            # Get bbox in XYWH format
            x, y, w, h = ann['bbox']
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Convert to XYXY
            x2, y2 = x + w, y + h
            
            # Normalize coordinates to [0, 1]
            x1_norm = x / img_w
            y1_norm = y / img_h
            x2_norm = x2 / img_w
            y2_norm = y2 / img_h
            
            # Clip to valid range
            x1_norm = max(0, min(1, x1_norm))
            y1_norm = max(0, min(1, y1_norm))
            x2_norm = max(0, min(1, x2_norm))
            y2_norm = max(0, min(1, y2_norm))
            
            # Skip degenerate boxes
            if x2_norm <= x1_norm or y2_norm <= y1_norm:
                continue
            
            boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            labels.append(self.coco_id_to_class[ann['category_id']])
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, boxes, labels, img_id
    
    def get_img_info(self, idx):
        """Get image information"""
        img_id = self.img_ids[idx]
        return self.coco.loadImgs(img_id)[0]


def collate_fn(batch):
    """
    Custom collate function for variable number of boxes per image
    
    Args:
        batch: List of (image, boxes, labels, img_id) tuples
    
    Returns:
        images: Tensor [B, 3, H, W]
        boxes: List of [N_i, 4] tensors
        labels: List of [N_i] tensors
        img_ids: List of image IDs
    """
    images, boxes, labels, img_ids = zip(*batch)
    
    # Stack images into batch
    images = torch.stack(images, 0)
    
    # Keep boxes and labels as lists (different lengths per image)
    return images, list(boxes), list(labels), list(img_ids)