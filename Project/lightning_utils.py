from imutils import paths
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import os
import pandas as pd
from torchvision.datasets.flickr import glob
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset, DataLoader
import torch
import lightning as L
import pandas as pd
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.segmentation import MeanIoU
import io
import matplotlib.pyplot as plt

class SegImageDataset(Dataset):
    def __init__(self, imgs, masks):
        self.imgs = imgs
        self.masks = masks
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = read_image(self.imgs[idx], ImageReadMode.RGB).float() / 255.0
        mask = torch.argmax(read_image(self.masks[idx], ImageReadMode.GRAY_ALPHA), dim=0).long()
        
        if not (image.shape[0] == mask.shape[0] and image.shape[0] == mask.shape[0]):
            image = resize(image, mask.shape)
        # binary_mask = torch.where(masked < 0.5, torch.tensor(0), torch.tensor(1))
        return image, mask
    
class SegDM(L.LightningDataModule):
    def __init__(self, batch_size, img_dir, mask_dir):
        super().__init__()
        # read and sort images.
        self.image_paths = sorted(list(paths.list_images(img_dir)))
        self.mask_paths = sorted(list(paths.list_images(mask_dir)))
        self.batch_size = batch_size
    
    def setup(self, stage: str):
        fit_imgs, test_imgs, fit_masks, test_masks = train_test_split(self.image_paths, self.mask_paths, test_size=0.25, random_state=42)
        if stage == "fit":
            train_imgs, val_imgs, train_masks, val_masks = train_test_split(fit_imgs, fit_masks, test_size=0.25, random_state=42)
            self.train = SegImageDataset(imgs=train_imgs, masks=train_masks)
            self.val = SegImageDataset(imgs=val_imgs, masks=val_masks)
            print(f"{len(self.train)} examples in the training set...")
            print(f"{len(self.val)} examples in the validation set...")
        if stage == "test":
            self.test = SegImageDataset(imgs=test_imgs, masks=test_masks)
            print(f"{len(self.test)} examples in the test set...")

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=0)
    

class SegModule(L.LightningModule):
    def __init__(self, model, num_classes, learning_rate=0.001, result_path='test_result.csv'):
        """
        model: any pytorch model
        num_classes: number of labels for pixels
        result_path: csv file path for test result
        """
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = learning_rate 
        self.result_path=result_path
        self.save_hyperparameters(ignore=['model', 'result_path'])
        
        self.test_results = {'f1': [], 'accuracy': [], 'precision': [], 'recall': [], 'mean_iou': []}
        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        self.mean_iou = MeanIoU(num_classes)
    
    def common_steps(self, batch):
        imgs, masks = batch
        preds = self.model(imgs)
        loss = self.loss_fn(preds, masks)
        return loss, preds, masks

    def training_step(self, batch):
        loss, _, _ = self.common_steps(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        loss, _, _ = self.common_steps(batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch):
        loss, preds, masks = self.common_steps(batch)
        self.log('test_loss', loss)
        preds = torch.argmax(preds, dim=1)
        # masks = torch.argmax(masks, dim=1)
        for metric_name in self.test_results.keys():
            metric = getattr(self, metric_name)
            self.test_results[metric_name].append(
                metric(preds, masks).item()
            )
        return loss
    
    def predict_step(self, test_image):
        test_image = test_image.unsqueeze(0)
        # Generate prediction
        output = self.model(test_image)
        prediction = torch.argmax(output, 0)
        return prediction
    
    def on_test_end(self):
        df = pd.DataFrame(self.test_results)
        df.to_csv(self.result_path, index=None)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
def get_segmentation_plot(trained_model, test_data, device, start_idx=0, end_idx=3):
    figs = []

    for i in range(start_idx, end_idx):
        test_image, test_mask = test_data[i]
        out = trained_model(test_image.to(device).unsqueeze(0))[0]
        prediction = torch.argmax(out, 0)

        fig, ax = plt.subplots(1, 3, figsize=(12, 8))
        ax[0].imshow(test_image.permute(1, 2, 0))
        ax[1].imshow(test_mask, cmap='gray', vmin=0, vmax=1, origin='lower')
        ax[2].imshow(prediction.detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')

        ax[0].set_title('Test Image')
        ax[1].set_title('Actual Mask')
        ax[2].set_title('Predicted Mask')

        figs.append(fig)

    return figs