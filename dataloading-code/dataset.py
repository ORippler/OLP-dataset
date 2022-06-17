import os
from typing import Dict, Optional, Callable, Any, Tuple
import cv2
import numpy as np
from torch.functional import Tensor
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
from albumentations import Compose, BboxParams
from pycocotools.coco import COCO
import torch
from multiprocessing import Pool


class FabricDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        load_mode: str = "both",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache: bool = False,
        uncached_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )

        self.coco = COCO(annFile)
        self.samples = list(sorted(self.coco.imgs.keys()))
        self.cats = self.coco.loadCats(self.coco.getCatIds())

        if load_mode not in ("both", "fl", "bl"):
            raise ValueError(
                f"Unknown load mode was specified, was: {load_mode}"
            )
        self.load_mode = load_mode
        if self.load_mode == "both":
            self._additional_target = {"image_bl": "image"}
        else:
            self._additional_target = {}

        self.all_classes = [cat["name"] for cat in self.cats]
        self.cache = cache
        self.uncached_transform = uncached_transform

        if self.cache:
            with Pool(8) as p:
                self.samples = p.map(self.load_transform, tqdm(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index
        Returns:
            A dictionary containing the following key/value pairs:
                "image":            loaded and transformed image
                "mask":             semantic segmentation mask of the image
                "masks":            binary segmentation masks of all individual defect-instances
                "bboxes":           bounding boxes in coco-format of the individual defect-instances
                "instance_labels":  class labels of the individual defect-instances
                "textile":          the fabric the image belongs to
                "target":           the, possibly transformed, image-level class-label/target
        """
        item = self.samples[index]
        if not self.cache:
            item = self.load_transform(item)

        # Transform mask and image at the same time (same random transform).
        if self.uncached_transform is not None:
            if (
                self._additional_target
                is not self.uncached_transform.additional_targets
            ):
                # TODO: Currently this step is performed every single time, would
                # be nicer to do it only once (even though it does not take long)
                self.uncached_transform = Compose(
                    self.uncached_transform.transforms,
                    additional_targets=self._additional_target,
                    bbox_params=BboxParams(
                        format="coco", label_fields=["instance_labels"]
                    ),
                )
            item = self.uncached_transform(**item)

        # Concatenate fl and bl images. For the bl image, we take the red
        # channel as opposed to calculating dedicated luminance values
        # as some of the images were acquired with RED LED only
        if self.load_mode == "both":
            if isinstance(item["image"], torch.Tensor):
                item["image"] = torch.cat(
                    (item["image"], item["image_bl"][0].unsqueeze(0)),
                    axis=0,
                )
            else:
                item["image"] = np.concatenate(
                    (
                        item["image"],
                        item["image_bl"][:, :, 0][..., np.newaxis],
                    ),
                    axis=-1,
                )
            del item["image_bl"]

        item["mask"] = item["masks"].pop(0)
        return item

    def load_transform(self, id: int) -> Dict[str, np.ndarray]:
        image_dict = self.coco.loadImgs(id)[0]
        if self.load_mode == "both":
            path_fl = image_dict["path_fl"]
            image = self._load_img(os.path.join(self.root, path_fl))
            path_bl = image_dict["path_bl"]
            image_bl = self._load_img(os.path.join(self.root, path_bl))
        else:
            image_path = image_dict["path_" + self.load_mode]
            image = self._load_img(os.path.join(self.root, image_path))
        if image is None or image.size == 0:
            raise OSError(f"Could not read image: {image_dict}")
        item = {"image": image}

        if self.load_mode == "both":
            item["image_bl"] = image_bl

        masks, instance_labels, bboxes = self.load_mask(id)
        item["masks"] = masks
        item["instance_labels"] = instance_labels
        item["bboxes"] = bboxes

        if self.transform is not None:
            if (
                self._additional_target
                is not self.transform.additional_targets
            ):
                self.transform = Compose(
                    self.transform.transforms,
                    additional_targets=self._additional_target,
                    bbox_params=BboxParams(
                        format="coco", label_fields=["instance_labels"]
                    ),
                )

            item = self.transform(**item)
        item["target"] = image_dict["target"]
        item["textile"] = image_dict["textile"]
        if self.target_transform is not None:
            item["target"] = self.target_transform(item["target"])
        return item

    def load_mask(self, id: int) -> np.ndarray:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        img = self.coco.loadImgs(id)[0]

        sem_seg_mask = np.zeros((img["height"], img["width"]), dtype=np.uint8)
        masks, instance_labels, bboxes = [], [], []
        for ann in anns:
            # need to wrap this since it is encoded in-place => will throw error
            # if sample is loaded twice otherwise
            if isinstance(ann["segmentation"]["counts"], str):
                ann["segmentation"]["counts"] = ann["segmentation"][
                    "counts"
                ].encode("UTF-8")
            catname, _ = self.getCatName(ann["category_id"])

            pixel_value = self.coco.getCatIds(catname)[
                0
            ]  # getCatIds returns a list

            instance_mask = self.coco.annToMask(ann)
            masks.append(instance_mask)
            bboxes.append(ann["bbox"])
            instance_labels.append(pixel_value)

            # we don't allow for overlapping instances (note that our gt also doesn't have overlapping instances anyway)
            sem_seg_mask = np.maximum(
                instance_mask * pixel_value, sem_seg_mask
            )

        masks.insert(
            0, sem_seg_mask
        )  # need to prepend due to albumentations bug: https://github.com/albumentations-team/albumentations/issues/1192

        # Use this to visualize masks via COCO
        # if anns:
        #     import random
        #     from matplotlib import pyplot as plt
        #     _rand = random.randint(0, 1000)
        #     for name in ("path_fl","path_bl"):
        #         image = self._load_img(os.path.join(self.root, img[name]))
        #         fig = plt.figure()
        #         fig.set_size_inches((img["width"]/500, img["height"]/500))
        #         ax = plt.Axes(fig, [0., 0., 1., 1.])
        #         ax.set_axis_off()
        #         fig.add_axes(ax)
        #         ax.imshow(image)
        #         np.random.seed(_rand)
        #         self.coco.showAnns(anns, draw_bbox=True)
        #         fig.savefig(f"../dataset_images/{img['textile']}_{img['id']}_{name.strip('path_')}.png")
        #     plt.show()
        return masks, instance_labels, bboxes

    def getCatName(self, classID: int) -> Tuple[str, str]:
        for cat in self.cats:
            if cat["id"] == classID:
                return cat["name"], cat["supercategory"]

    @staticmethod
    def _load_img(
        path: str, mode: int = cv2.IMREAD_COLOR, dtype: np.dtype = np.uint8
    ) -> np.ndarray:
        img = cv2.imread(path, mode)
        if mode is cv2.IMREAD_COLOR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode is cv2.IMREAD_GRAYSCALE:
            img = np.expand_dims(img, axis=-1)
        img = img.astype(dtype, copy=False)
        return img
