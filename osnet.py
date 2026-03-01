import numpy as np
import torch
import cv2

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False

# OSNet expects (height, width) = (256, 128), ImageNet normalization
INPUT_H, INPUT_W = 256, 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model = None
_device = None


def _get_model():
    """Lazy-load OSNet once (pretrained on ImageNet; good for Re-ID features)."""
    global _model, _device
    if _model is not None:
        return _model, _device
    if not TORCHREID_AVAILABLE:
        raise ImportError(
            "torchreid is required for OSNet. Install with: pip install torchreid"
        )
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1,
        pretrained=True,
        loss="softmax",
    )
    _model = _model.to(_device)
    _model.eval()
    return _model, _device


def _preprocess(crop_bgr):
    """
    crop_bgr: numpy array (H, W, 3) BGR, uint8.
    Returns: tensor (1, 3, INPUT_H, INPUT_W).
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return torch.from_numpy(x).float()


def osnet_vector(crop_bgr):
    """
    Extract OSNet Re-ID feature vector from a person crop.

    Args:
        crop_bgr: numpy array (H, W, 3) BGR (e.g. from cv2), uint8.
                  Will be resized to 256x128.

    Returns:
        1D numpy array of shape (512,) dtype float32. Use with cosine similarity
        (e.g. compareVector in main.py) for matching across frames.
    """
    model, device = _get_model()
    x = _preprocess(crop_bgr).to(device)
    with torch.no_grad():
        # In eval mode, torchreid OSNet returns the 512-d feature vector, not logits.
        features = model(x)
    out = features.cpu().numpy().flatten().astype(np.float32)
    return out


def osnet_vector_from_file(path):
    """Load image from path (BGR) and return OSNet feature vector."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return osnet_vector(img)
