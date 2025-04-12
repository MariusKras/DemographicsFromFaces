"""Functions and classes to make and evaluate CNN predictions"""

import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union, Any
from torch import nn
from lime import lime_image
from skimage import color


class UTKFaceDataset(Dataset):
    """Dataset for UTKFace images."""

    def __init__(self, df: pd.DataFrame, transform: Optional[Any] = None) -> None:
        """Initialize UTKFaceDataset with DataFrame and optional transform."""
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.transform: Optional[Any] = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Union[Tensor, PILImage], Tensor, Tensor]:
        """Return image, age, and gender for a given index."""
        row = self.df.iloc[idx]
        img_path: str = row["image_path"]
        image: PILImage = PILImage.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        age: Tensor = torch.tensor(row["Age"], dtype=torch.float32).unsqueeze(0)
        gender: Tensor = torch.tensor(row["Gender"], dtype=torch.long)
        return image, age, gender


def plot_images_probability(
    df: pd.DataFrame, suptitle: str, num_rows_display: int
) -> None:
    """Plot images from a DataFrame in a grid with a specified title and number of display rows."""
    num_rows = min(num_rows_display, -(-len(df) // 3))
    plt.figure(figsize=(12, num_rows * 4))
    plt.suptitle(suptitle, fontsize=14, y=0.985)
    for i, row in enumerate(df.itertuples()):
        if i >= num_rows * 3:
            break
        img = PILImage.open(row.image_path)
        plt.subplot(num_rows, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Age: {row.Age}, Pred Prob of Being Female: {row._6:.2f}")
    plt.tight_layout()
    plt.show()


def denormalize_image(img_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a normalized tensor image to [0, 1] range."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = img_tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)
    return img_tensor.clamp(0, 1)


def lime_plots_multitask(
    model: nn.Module, test_loader: Any, number_of_images: int
) -> None:
    """Generate LIME explanations for multi-task model."""
    model.eval()
    device = next(model.parameters()).device

    def batch_predict_gender(images: np.ndarray) -> np.ndarray:
        tensor_images = (
            torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(device)
        )
        with torch.no_grad():
            age_preds, gender_logits = model(tensor_images)
            probs = torch.softmax(gender_logits, dim=1)
        return probs.cpu().numpy()

    def batch_predict_age(images: np.ndarray) -> np.ndarray:
        tensor_images = (
            torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(device)
        )
        with torch.no_grad():
            age_preds, gender_logits = model(tensor_images)
        return age_preds.cpu().numpy()

    images, true_ages, true_genders = next(iter(test_loader))
    images = images[:number_of_images]
    true_ages = true_ages[:number_of_images].numpy().flatten()
    true_genders = true_genders[:number_of_images].numpy()
    lime_explainer = lime_image.LimeImageExplainer()
    results = []
    for i, img_tensor in enumerate(images):
        img = denormalize_image(img_tensor).permute(1, 2, 0).cpu().numpy()
        explanation_gender = lime_explainer.explain_instance(
            img, batch_predict_gender, top_labels=1, hide_color=0, num_samples=6000
        )
        top_label_gender = explanation_gender.top_labels[0]
        _, gender_mask = explanation_gender.get_image_and_mask(
            top_label_gender, positive_only=False, num_features=5, hide_rest=False
        )
        predicted_gender_probs = batch_predict_gender(np.array([img]))[0]
        predicted_gender = top_label_gender
        predicted_gender_prob = predicted_gender_probs[top_label_gender]
        explanation_age = lime_explainer.explain_instance(
            img, batch_predict_age, num_samples=6000, hide_color=0
        )
        _, age_mask = explanation_age.get_image_and_mask(
            0, positive_only=False, num_features=5, hide_rest=False
        )
        predicted_age = batch_predict_age(np.array([img]))[0][0]
        colors = ["cyan", "red"]
        colored_mask_gender = color.label2rgb(
            gender_mask, img, alpha=0.5, bg_label=0, colors=colors
        )
        colored_mask_age = color.label2rgb(
            age_mask, img, alpha=0.5, bg_label=0, colors=colors
        )
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        colored_mask_gender_uint8 = np.uint8(colored_mask_gender * 255)
        colored_mask_age_uint8 = np.uint8(colored_mask_age * 255)
        blended_gender = cv2.addWeighted(
            img_uint8, 0.4, colored_mask_gender_uint8, 0.9, -5
        )
        blended_age = cv2.addWeighted(img_uint8, 0.4, colored_mask_age_uint8, 0.9, -5)
        results.append(
            (
                blended_gender,
                true_genders[i],
                predicted_gender,
                predicted_gender_prob,
                blended_age,
                true_ages[i],
                predicted_age,
            )
        )
    num_images = len(results)
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
    fig.suptitle("LIME Explanations", fontsize=14, y=0.99)
    plt.subplots_adjust(wspace=0.05, hspace=0.2, top=0.95)
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)
    for i, (
        img_gender,
        true_gender,
        pred_gender,
        prob_gender,
        img_age,
        true_age,
        pred_age,
    ) in enumerate(results):
        axes[i, 0].imshow(img_gender)
        axes[i, 0].set_title(
            f"True Gender: {true_gender}, Pred Gender: {pred_gender}\nProb of Being Female: {prob_gender:.2f}",
            fontsize=12,
        )
        axes[i, 0].axis("off")
        axes[i, 1].imshow(img_age)
        axes[i, 1].set_title(
            f"True Age: {true_age:.1f}, Pred Age: {pred_age:.1f}", fontsize=12
        )
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()
