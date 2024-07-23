from imutils import paths
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset, DataLoader
import torch
import lightning as L
import pandas as pd
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.segmentation import MeanIoU
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import CSVLogger
import os

class SegImageDataset(Dataset):
    def __init__(self, imgs, masks):
        self.imgs = imgs
        self.masks = masks
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = read_image(self.imgs[idx], ImageReadMode.RGB).float() / 255.0
        mask = torch.argmax(read_image(self.masks[idx], ImageReadMode.GRAY_ALPHA), dim=0).long()

        if not (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]):
            image = resize(image, mask.shape, antialias=True)
        
        return image, mask

class SegDM(L.LightningDataModule):
    def __init__(self, batch_size, img_dir, mask_dir, collate_fn=None, num_workers = 4):
        super().__init__()
        # read and sort images.
        self.image_paths = sorted(list(paths.list_images(img_dir)))
        self.mask_paths = sorted(list(paths.list_images(mask_dir)))
        self.batch_size = batch_size
        self.num_workers = num_workers 
        self.collate_fn = collate_fn
    
    def setup(self, stage: str):
        fit_imgs, test_imgs, fit_masks, test_masks = train_test_split(
            self.image_paths, self.mask_paths, test_size=0.25, random_state=42
        )
        if stage == "fit":
            train_imgs, val_imgs, train_masks, val_masks = train_test_split(
                fit_imgs, fit_masks, 
                test_size=0.25, 
                random_state=42
            )
            self.train = SegImageDataset(imgs=train_imgs, masks=train_masks)
            self.val = SegImageDataset(imgs=val_imgs, masks=val_masks)
            print(f"{len(self.train)} examples in the training set...")
            print(f"{len(self.val)} examples in the validation set...")
        
        if stage == "test":
            self.test = SegImageDataset(imgs=test_imgs, masks=test_masks)
            print(f"{len(self.test)} examples in the test set...")

    def train_dataloader(self):
        return DataLoader(self.train, 
                          shuffle=True, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val, 
                          shuffle=False, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test, 
                          shuffle=False, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)
    

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
        self.num_classes = num_classes
        self.test_results = {'f1': [], 'accuracy': [], 'precision': [], 'recall': [], 'mean_iou': []}
        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        self.mean_iou = MeanIoU(self.num_classes)

        self.save_hyperparameters(ignore=['model', 'result_path'])
    
    def common_steps(self, batch):
        imgs, masks = batch
        preds = self.model(imgs)
        loss = self.loss_fn(preds, masks)
        return loss, preds, masks

    def training_step(self, batch):
        loss, _, _ = self.common_steps(batch)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        loss, _, _ = self.common_steps(batch)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss
    
    def test_step(self, batch):
        loss, preds, masks = self.common_steps(batch)
        self.log('test_loss', loss, sync_dist=True)
        preds = torch.argmax(preds, dim=1)
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
        print('f1:', df['f1'].mean())
        print('accuracy:', df['accuracy'].mean())
        print('precision:', df['precision'].mean())
        print('recall:', df['recall'].mean())
        print('mean_iou:', df['mean_iou'].mean())
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
def get_segmentation_plot(
        trained_model: torch.nn.Module, 
        test_data, 
        device, 
        start_idx=0):
    """
    Args:
        trained_model: any pytorch model
        test_data: dataset of type torch.utils.data.Dataset
        device: cpu/mps/gpu
    Return:
        3 plots from matplotlib
    """
    figs = []
    trained_model.to(torch.device(device))
    trained_model.eval()
    end_idx = start_idx + 3
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

kaggle_paths = dict(
    darwin=dict(
        img_dir='/kaggle/input/xray-segmentation/Darwin/Darwin/img', 
        mask_dir='/kaggle/input/xray-segmentation/Darwin/Darwin/mask'
    ),
    shenzen=dict(
        img_dir='/kaggle/input/xray-segmentation/Shenzhen/Shenzhen/img', 
        mask_dir='/kaggle/input/xray-segmentation/Shenzhen/Shenzhen/mask'
    ),
    covid=dict(
        img_dir='/kaggle/input/xray-segmentation/Covid19 Radiography/COVID-19_Radiography_Dataset/COVID/images', 
        mask_dir='/kaggle/input/xray-segmentation/Covid19 Radiography/COVID-19_Radiography_Dataset/COVID/masks'
    ),
)

local_paths = dict(
    darwin=dict(
        img_dir='./datasets/Darwin/img', 
        mask_dir='./datasets/Darwin/mask'
    ),
    shenzen=dict(
        img_dir='./datasets/Shenzhen/img', 
        mask_dir='./datasets/Shenzhen/mask'
    ),
    covid=dict(
        img_dir='./datasets/COVID-19_Radiography_Dataset/COVID/images', 
        mask_dir='./datasets/COVID-19_Radiography_Dataset/COVID/masks'
    ),
)


def get_seg_lightning_modules(data_paths, 
                              model, 
                              model_name, 
                              collate_fn=None,
                              accelerator='mps', 
                              fast=True, 
                              ckpt=None,
                              batch_size=2,
                              devices=[0, 1], # For gpu
                              max_epochs=10, 
                              n_classes=2, 
                              learning_rate=0.001):
    """
    Agrs:
        - data_paths: Dictionary to mask and imgs datasets
        - model: Pytorch Model Object
        - ckpt: path to checkpoint folder
        - model_name: for naming /logs/ folder, result csv filename.
        - accelerator=mps: cpu, mps, or gpu
        - fast=True: for testing pipeline
        - batch_size=2
        - max_epochs = 10
        - learning_rate = 0.001
        - n_classes = 2
        - collate_fn=None: 
            - process the batch of image of size (batch_size, channels, H, W) before passing it to the model
            - this can be used for the pre-process steps of the HuggingFace models
    Returns:
        - DataModule
        - Module
        - Trainer
    """
    data_module = SegDM(batch_size=batch_size, 
                        collate_fn=collate_fn, 
                        **data_paths)
    logger = CSVLogger("logs", name=model_name)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if ckpt:
        module = SegModule.load_from_checkpoint(model, 
                           num_classes=n_classes,
                           result_path=f'./results/{model_name}.csv', 
                           learning_rate=learning_rate,
                           checkpoint_path=ckpt)
    else:
        module = SegModule(model, 
                           num_classes=n_classes,
                           result_path=f'./results/{model_name}.csv', 
                           learning_rate=learning_rate)
    
    if accelerator == "mps" or accelerator == "cpu":
        trainer = L.Trainer(fast_dev_run=fast, 
                            logger=logger, 
                            max_epochs=max_epochs)
    else:
        trainer = L.Trainer(fast_dev_run=fast, 
                            logger=logger, accelerator="gpu",
                            devices=devices, 
                            max_epochs=max_epochs)
    
    return data_module, module, trainer