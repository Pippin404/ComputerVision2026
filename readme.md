## Task
- Train and evaluate a ResNet-18 model on seven-segment digit images.
- Support three modes: train, test, and predict.

## Data
- Image dataset of seven-segment digit images.
- Label is taken from the first character of each filename.
- Supported image formats: PNG, JPG, JPEG, BMP, TIF, TIFF, WEBP.
- Images are loaded as grayscale, then converted to RGB for ResNet-18.
- Training data is limited to a maximum of 50 images per label.
- If no validation folder is available, the script splits the training set into train and validation subsets.

## Model Info (ResNet-18)
- Base architecture: ResNet-18 from torchvision.
- Final fully connected layer is replaced to match the number of classes in the dataset.
- Optional ImageNet pretrained weights can be enabled.
- Input images are resized to 224 x 224 by default.
- Normalization uses mean = [0.5, 0.5, 0.5] and std = [0.5, 0.5, 0.5].

## Approach (Epochs, Class Size)
- Train for 15 epochs by default.
- Batch size is 32 by default.
- Optimizer: Adam with learning rate 0.001.
- Data augmentation on training images uses random affine transforms.
- Class size is capped at 50 images per label for training balance.
- Best model checkpoint is saved when validation accuracy improves.

## Early Results
- Validation loss and accuracy are tracked after each epoch.
- The best validation accuracy is stored as the checkpoint criterion.
- Test mode loads the saved checkpoint and reports loss and accuracy.
- Prediction mode outputs the predicted digit and confidence for one image.
- No numeric results are present in the script itself; these need to be filled in after running training.

## Additional Info
- The script uses a fixed random seed for reproducibility.
- It automatically selects CUDA if available, otherwise CPU.
- Checkpoints save class mappings, model weights, and image size.
- The input image path can be supplied for single-image prediction.
