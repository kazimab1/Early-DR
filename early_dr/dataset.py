"""Dataset, augmentation and the samplers used to handle class imbalance."""
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .data import task_label
from .preprocess import processed_path

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class AddNoise:
    """Slight gaussian noise on a tensor image."""

    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma


def colour_normalise(img):
    """'graham': subtract the local average colour to remove illumination differences."""
    blur = cv2.GaussianBlur(img, (0, 0), img.shape[0] / 30)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def build_transform(kind, size):
    """mild = small flips/rotation/colour jitter.  strong = rotations, crop+zoom, blur, noise."""
    if kind == "strong":
        steps = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
                 T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                 T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
                 T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
                 T.ToTensor(), AddNoise(), T.Normalize(MEAN, STD)]
    elif kind == "mild":
        steps = [T.RandomHorizontalFlip(), T.RandomRotation(15),
                 T.ColorJitter(brightness=0.1, contrast=0.1), T.ToTensor(), T.Normalize(MEAN, STD)]
    else:                                              # "none": used for validation and test
        steps = [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(steps)


class DRDataset(Dataset):
    """Returns (image tensor, label, row number). `labels` are already task labels."""

    def __init__(self, df, cfg, labels, train=False):
        self.paths = [processed_path(cfg, i) for i in df.image_id]
        self.labels = np.asarray(labels)
        self.size = cfg.model.image_size
        self.colour_norm = cfg.colour_norm
        self.train = train
        self.aug_scope = cfg.imbalance.aug_scope
        self.majority = np.bincount(self.labels).argmax() if len(self.labels) else 0
        self.plain = build_transform("none", self.size)
        self.mild = build_transform("mild", self.size)
        self.strong = build_transform(cfg.imbalance.aug, self.size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        if self.colour_norm == "graham":
            img = colour_normalise(img)
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(img)
        label = int(self.labels[i])
        if not self.train:
            transform = self.plain
        elif self.aug_scope == "minority" and label == self.majority:
            transform = self.mild                      # the majority class only gets the mild policy
        else:
            transform = self.strong
        return transform(img), label, i


# ----- samplers ---------------------------------------------------------------------------------

def source_factor(df, grades):
    """Weight that makes every source equally likely INSIDE each grade (used by arm B).
    The total weight of each grade is unchanged, so grade frequencies stay natural."""
    factor = np.ones(len(df))
    for g in np.unique(grades):
        in_grade = grades == g
        sources = df.source.values[in_grade]
        names, counts = np.unique(sources, return_counts=True)
        for name, count in zip(names, counts):
            factor[in_grade & (df.source.values == name)] = in_grade.sum() / len(names) / count
    return factor


def class_factor(labels, n_classes, cfg):
    """How much more likely each image of a class is to be drawn, for each sampler type."""
    counts = np.bincount(labels, minlength=n_classes).clip(min=1)
    mode = cfg.imbalance.sampler
    if mode in ("weighted", "class_aware"):
        return 1.0 / counts
    if mode == "repeat_factor":
        freq = counts / counts.sum()
        repeat = np.sqrt(cfg.imbalance.repeat_threshold / freq)
        return np.clip(repeat, 1.0, cfg.imbalance.repeat_cap)
    return np.ones(n_classes)


class ClassAwareBatches:
    """Every mini-batch holds the same number of images from each class."""

    def __init__(self, labels, weights, batch_size, seed=0):
        self.groups = [np.where(labels == c)[0] for c in np.unique(labels)]
        self.probs = [weights[g] / weights[g].sum() for g in self.groups]
        self.batch_size, self.n_batches = batch_size, len(labels) // batch_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        per_class = int(np.ceil(self.batch_size / len(self.groups)))
        for _ in range(self.n_batches):
            batch = np.concatenate([self.rng.choice(g, per_class, p=p)
                                    for g, p in zip(self.groups, self.probs)])
            self.rng.shuffle(batch)
            yield batch[:self.batch_size].tolist()


def make_sampler(df, labels, n_classes, cfg):
    """Returns (DataLoader keyword arguments, effective class prior seen during training).
    The prior is needed later to correct probabilities back to natural prevalence."""
    grades = df.grade.values
    per_image = class_factor(labels, n_classes, cfg)[labels]
    if cfg.data.arm == "B":
        per_image = per_image * source_factor(df, grades)

    prior = np.bincount(labels, weights=per_image, minlength=n_classes)
    prior = prior / prior.sum()

    if cfg.imbalance.sampler == "class_aware":
        batches = ClassAwareBatches(labels, per_image, cfg.train.batch_size, cfg.seed)
        return {"batch_sampler": batches}, np.full(n_classes, 1.0 / n_classes)
    if cfg.imbalance.sampler == "none" and cfg.data.arm != "B":
        return {"shuffle": True, "batch_size": cfg.train.batch_size, "drop_last": True}, prior
    n_draws = len(labels)
    if cfg.imbalance.sampler == "repeat_factor":       # repeating rare images makes the epoch longer
        n_draws = int(class_factor(labels, n_classes, cfg)[labels].sum())
    sampler = WeightedRandomSampler(per_image, num_samples=n_draws, replacement=True)
    return {"sampler": sampler, "batch_size": cfg.train.batch_size, "drop_last": True}, prior
