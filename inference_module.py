import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import joblib
import json
import warnings

IMG_SIZE = 224

def preprocess_image(img, size=IMG_SIZE, augment=False):
    if img is None:
        return np.full((size, size), 255, dtype=np.uint8)
    img = np.asarray(img, dtype=np.uint8)

    ys, xs = np.where(img < 200)
    if len(xs) == 0:
        return np.full((size, size), 255, dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = img[y_min:y_max+1, x_min:x_max+1]

    h, w = cropped.shape
    scale = size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size), 255, dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)

    return canvas

def gradient_orientation_histogram(img, num_bins=9):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    bin_width = 180 / num_bins
    hist = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mask = (angle >= i * bin_width) & (angle < (i + 1) * bin_width)
        hist[i] = np.sum(magnitude[mask])
    hist /= (np.linalg.norm(hist) + 1e-6)
    return hist

def grid_density(img, grid=7):
    h, w = img.shape
    gh, gw = h // grid, w // grid
    features = []
    for i in range(grid):
        for j in range(grid):
            cell = img[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            features.append(np.mean(cell > 0))
    return np.array(features, dtype=np.float32)

def gradient_magnitude_stats(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return np.array([
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.percentile(mag, 75)
    ], dtype=np.float32)

def ink_ratio_feature(img):
    return np.array([np.mean(img > 0)], dtype=np.float32)

def hv_stroke_ratio(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    abs_gx = np.mean(np.abs(gx))
    abs_gy = np.mean(np.abs(gy))
    return np.array([abs_gx, abs_gy, abs_gx / (abs_gy + 1e-6)], dtype=np.float32)

def quadrant_density(img):
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    return np.array([
        np.mean(img[:h2, :w2] > 0),
        np.mean(img[:h2, w2:] > 0),
        np.mean(img[h2:, :w2] > 0),
        np.mean(img[h2:, w2:] > 0)
    ], dtype=np.float32)

def edge_density_feature(img):
    edges = cv2.Canny(img, 50, 150)
    return np.array([np.mean(edges > 0)], dtype=np.float32)

def normalized_moment_features(img):
    img = img.astype(np.float32)
    total = img.sum() + 1e-6
    y, x = np.indices(img.shape)
    x_bar = (x * img).sum() / total
    y_bar = (y * img).sum() / total
    mu20 = ((x - x_bar)**2 * img).sum() / total
    mu02 = ((y - y_bar)**2 * img).sum() / total
    mu11 = ((x - x_bar)*(y - y_bar) * img).sum() / total
    return np.array([mu20, mu02, mu11], dtype=np.float32)

def shape_ratio_features(img):
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return np.zeros(3, dtype=np.float32)
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    area = len(xs)
    aspect = width / (height + 1e-6)
    compactness = area / (width * height + 1e-6)
    return np.array([aspect, 1.0 / (aspect + 1e-6), compactness], dtype=np.float32)

def extract_features_manual(img):
    feats = []

    feats.append(gradient_orientation_histogram(img))

    def spatial_orientation(img_in, grid):
        h, w = img_in.shape
        cell_h = h // grid
        cell_w = w // grid
        local = []
        for i in range(grid):
            for j in range(grid):
                patch = img_in[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                local.append(gradient_orientation_histogram(patch))
        return np.concatenate(local)

    feats.append(spatial_orientation(img, grid=2))
    feats.append(spatial_orientation(img, grid=4))

    feats.append(grid_density(img, grid=5))
    feats.append(grid_density(img, grid=7))

    feats.append(gradient_magnitude_stats(img))

    feats.append(ink_ratio_feature(img))
    feats.append(hv_stroke_ratio(img))
    feats.append(quadrant_density(img))
    feats.append(edge_density_feature(img))

    feats.append(normalized_moment_features(img))
    feats.append(shape_ratio_features(img))

    return np.concatenate(feats)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        return self.out(x) / 1.5

class DrawPredictor:
    def __init__(self, model_path="quickdraw_mlp.pth", scaler_path="scaler.pkl", map_path="class_map.json"):
        with open(map_path, "r") as f:
            self.idx_to_cls = {v: k for k, v in json.load(f).items()}
        self.class_names = [self.idx_to_cls[i] for i in sorted(self.idx_to_cls.keys())]
        self.scaler = joblib.load(scaler_path)
        input_dim = getattr(self.scaler, "n_features_in_", 138)
        self.model = MLPClassifier(input_dim=input_dim, num_classes=len(self.idx_to_cls))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def _align_features_dim(self, features: np.ndarray) -> np.ndarray:
        target_dim = getattr(self.model.fc1, "in_features", features.shape[1])
        current_dim = features.shape[1]
        if current_dim == target_dim:
            return features
        if current_dim > target_dim:
            return features[:, :target_dim]
        pad = target_dim - current_dim
        return np.concatenate([features, np.zeros((features.shape[0], pad), dtype=features.dtype)], axis=1)

    def _predict_features(self, features):
        features = features.reshape(1, -1)
        features = self._align_features_dim(features)
        try:
            scaler_dim = getattr(self.scaler, "n_features_in_", None)
            if scaler_dim is not None and scaler_dim == features.shape[1]:
                features_norm = self.scaler.transform(features)
            else:
                warnings.warn(
                    f"Scaler feature mismatch (scaler={scaler_dim}, features={features.shape[1]}). "
                    "Proceeding without scaling.")
                features_norm = features
        except Exception as e:
            warnings.warn(f"Feature scaling failed: {e}. Proceeding without scaling.")
            features_norm = features
        tensor_in = torch.tensor(features_norm, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def predict_array(self, img_array):
        img_pp = preprocess_image(img_array, augment=False)
        features = extract_features_manual(img_pp)
        probs = self._predict_features(features)
        idx = int(np.argmax(probs))
        return self.idx_to_cls[idx], float(probs[idx])

    def predict(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return self.predict_array(img)

    def predict_topk(self, img_array, k=3):
        img_pp = preprocess_image(img_array, augment=False)
        features = extract_features_manual(img_pp)
        probs = self._predict_features(features)
        topk_idx = np.argsort(probs)[::-1][:k]
        return [(self.idx_to_cls[int(i)], float(probs[int(i)])) for i in topk_idx]