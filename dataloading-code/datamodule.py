from typing import Callable
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from dataset import FabricDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import os
import torch
from typing import Any
import albumentations as A
from sklearn.utils.validation import indexable, _num_samples


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    needed_keys = {
        "image": torch.stack,
        "mask": torch.stack,
        "target": torch.tensor,
        "textile": torch.tensor,
    }
    needed_types = {
        "image": torch.float,
        "mask": torch.uint8,
        "target": torch.uint8,
        "textile": torch.uint8,
    }
    data = {}
    for key, fn in needed_keys.items():
        data[key] = fn([sample[key] for sample in batch])
        data[key] = data[key].type(needed_types[key])

    return data


def binary_target_transform(target: list[int]) -> int:
    """
    Default transform for Anomaly Detection (converts defect instances to the
    Binary problem)
    """
    if len(target) == 1 and target[0] == 0:
        return 0
    else:
        return 1


class FabricDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        textiles: list = [],
        invert_textiles: bool = False,  # whether to invert the textiles_list
        add_normal: bool = False,  # whether to add the normal data of the held out fabrics to the large-scale training & validation, and the anomalies ot the test set
        semi_supervised_split: bool = False,  # whether to use semi-supervised training/iteration (defect-free only for training + val, defects only in test)
        cache: bool = False,
        transform: Callable = A.Compose([A.NoOp()]),
        target_transform: Callable = binary_target_transform,
        uncached_train_transform: Callable = A.Compose([A.NoOp()]),
        uncached_eval_transform: Callable = A.Compose([A.NoOp()]),
        batch_size: int = 4,
        load_mode: str = "both",
        fold: int = 0,
        frac_supervised: float = 0.25,
        datasplit: str = "B",
        collate_fn: Callable = collate_fn,
    ) -> None:
        super().__init__()
        self.root = root
        self.invert_textiles = invert_textiles
        self.add_normal = add_normal
        self.semi_supervised_split = semi_supervised_split
        self.cache = cache
        self.target_transform = target_transform
        self.transform = transform
        self.uncached_train_transform = uncached_train_transform
        self.uncached_eval_transform = uncached_eval_transform
        self.batch_size = batch_size
        self.folds = 5  # let's hardcode this
        self.fold = fold
        self.frac_supervised = frac_supervised
        self.load_mode = load_mode
        self.datasplit = datasplit
        self.collate_fn = collate_fn

        for textile in textiles:
            assert isinstance(textile, int)
        self.textiles = sorted(textiles)

        if self.datasplit == "B":
            self.datasplit_textiles = list(range(1, 39))
        elif self.datasplit == "A":
            self.datasplit_textiles = list(range(1, 22))
        else:
            raise ValueError(
                f"Unknown configuration for datasplit. Must be in ['A', 'B'], was {self.datasplit}"
            )

        for textile in self.textiles:
            if textile not in self.datasplit_textiles:
                raise RuntimeError(
                    "All specified textiles must be in the provided datasplit"
                )

    def _setup_single(
        self, invert: bool, semi_supervised_split: bool
    ) -> tuple[Subset, Subset, Subset, list]:
        # Filter & sort textiles to ensure determinism
        folder_contents = [
            folder
            for folder in os.scandir(self.root)
            if "Textile_" in folder.name
        ]
        if invert:
            filtered_folder_contents = [
                folder
                for folder in folder_contents
                if int(folder.name.split("Textile_")[-1]) not in self.textiles
                and int(folder.name.split("Textile_")[-1])
                in self.datasplit_textiles
            ]
        else:
            filtered_folder_contents = [
                folder
                for folder in folder_contents
                if int(folder.name.split("Textile_")[-1]) in self.textiles
                and int(folder.name.split("Textile_")[-1])
                in self.datasplit_textiles
            ]
        sorted_root_folder = sorted(
            filtered_folder_contents,
            key=lambda x: (int(x.name.split("Textile_")[-1])),
        )

        datasets = [
            FabricDataset(
                root=folder.path,
                annFile=os.path.join(folder.path, "dataset.json"),
                cache=self.cache,
                transform=self.transform,
                target_transform=self.target_transform,
                uncached_transform=self.uncached_train_transform,
                load_mode=self.load_mode,
            )
            for folder in sorted_root_folder
        ]

        # complete_dataset holds both defects and defect_free data
        complete_dataset = ConcatDataset(datasets)

        targets_all, textiles_all = [], []
        for dataset in complete_dataset.datasets:
            for id in range(len(dataset)):
                sample = dataset.coco.loadImgs(id)[0]
                targets_all.append(sample["target"])
                textiles_all.append(sample["textile"])

        if semi_supervised_split:
            binary_targets = list(map(binary_target_transform, targets_all))
            binary_targets = torch.tensor(binary_targets, dtype=bool).squeeze()
            idx_all = np.arange(len(complete_dataset))
            idx_normal = idx_all[~binary_targets]
            idx_anomalous = idx_all[binary_targets]

            complete_set = Subset(complete_dataset, idx_normal)
            defective_images = Subset(complete_dataset, idx_anomalous)
            # need to construct dummy targets_sampled
            targets_sampled = np.zeros(len(complete_set))
            splitter = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        else:
            # We perform stratification for multi-label images by selecting a random single-class from them,
            # But we could also go for multi-label stratification by skmultilearn (one could also try to additionally)
            # Stratify for fabric prevalence by these means
            targets_sampled = [
                np.random.choice(target, size=1) for target in targets_all
            ]

            splitter = StratifiedKFold(
                n_splits=self.folds, shuffle=True, random_state=0
            )

            complete_set = complete_dataset

        # We perform stratified K-2/1/1 splits
        gen = self._three_way_split(
            splitter, np.zeros(len(complete_set)), targets_sampled
        )
        # iterate until the appropriate fold
        for _ in range(self.fold + 1):
            train_idx, val_idx, test_idx = next(gen)

        train_classes = [targets_sampled[index] for index in train_idx]
        train_dataset = Subset(complete_set, train_idx)
        val_dataset = Subset(complete_set, val_idx)
        test_dataset = Subset(complete_set, test_idx)

        if semi_supervised_split:
            test_dataset = ConcatDataset((test_dataset, defective_images))

        return train_dataset, val_dataset, test_dataset, train_classes

    def setup(self, stage: str = None) -> None:

        (
            train_dataset,
            val_dataset,
            test_dataset,
            train_classes,
        ) = self._setup_single(
            self.invert_textiles, self.semi_supervised_split
        )

        if self.add_normal:
            # we invert the textiles, and always apply the semi_supervised_split (since it's add_normal)
            (
                train_dataset_n,
                val_dataset_n,
                test_dataset_n,
                train_classes_n,
            ) = self._setup_single(not self.invert_textiles, True)

            train_dataset = ConcatDataset([train_dataset, train_dataset_n])
            val_dataset = ConcatDataset([val_dataset, val_dataset_n])
            test_dataset = ConcatDataset([test_dataset, test_dataset_n])
            train_classes.extend(train_classes_n)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        if self.frac_supervised != 0 and not self.semi_supervised_split:
            # compute sampling ratios
            weights_list = np.empty_like(
                train_classes, dtype=np.float32
            ).squeeze()
            a_samples = np.array(train_classes).squeeze() != 0
            w_a = self.frac_supervised / a_samples.sum()
            weights_list[a_samples] = w_a

            n_samples = np.array(train_classes).squeeze() == 0
            w_n = (1 - self.frac_supervised) / n_samples.sum()
            weights_list[n_samples] = w_n

            self.sampler = WeightedRandomSampler(
                weights_list, len(self.train_dataset)
            )
        else:
            self.sampler = RandomSampler(self.train_dataset)

    def train_dataloader(self, shuffle: bool = True):
        self.set_uncached_transform(
            self.train_dataset, self.uncached_train_transform
        )
        _num_workers = 8 if self.cache else 16
        _sampler = self.sampler if shuffle else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=_sampler,
            num_workers=_num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        self.set_uncached_transform(
            self.val_dataset, self.uncached_eval_transform
        )
        _num_workers = 4 if self.cache else 8
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            num_workers=_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        self.set_uncached_transform(
            self.test_dataset, self.uncached_eval_transform
        )
        _num_workers = 2
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            num_workers=_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    @staticmethod
    def set_uncached_transform(dset, uncached_transform):
        if isinstance(dset, FabricDataset):
            dset.uncached_transform = uncached_transform
        elif isinstance(dset, Subset):
            FabricDataModule.set_uncached_transform(
                dset.dataset, uncached_transform
            )
        elif isinstance(dset, ConcatDataset):
            for dataset in dset.datasets:
                FabricDataModule.set_uncached_transform(
                    dataset, uncached_transform
                )

    @staticmethod
    def _three_way_split(splitter, X, y=None, groups=None):
        """A modified version of BaseCrossValidator.split().
        Yields (K-2/1/1) train/val/test splits.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        test_masks_it = splitter._iter_test_masks(X, y, groups)
        first_mask = last_mask = next(test_masks_it)
        for test_mask in test_masks_it:
            train_index = indices[
                np.logical_not(np.logical_or(test_mask, last_mask))
            ]
            val_index = indices[last_mask]
            test_index = indices[test_mask]
            yield train_index, val_index, test_index
            last_mask = test_mask
        # last fold
        test_mask = first_mask
        train_index = indices[
            np.logical_not(np.logical_or(test_mask, last_mask))
        ]
        val_index = indices[last_mask]
        test_index = indices[test_mask]
        yield train_index, val_index, test_index
