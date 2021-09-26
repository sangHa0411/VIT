# VIT

# ImageNet Image Classification

## Modules
```
├── cutmix.py
├── dataset.py
├── loss.py
├── model.py
├── scheduler.py
└── train.py
```

## Model 
  1. VIT-Base/16
  
## Model Specification
  1. Total layer size : 12
  2. Input image size : 224
  3. Input channel size : 3
  4. Patch Size : 16
  5. Embedding Size : 768
  6. Hidden Size : 3072
  7. Class size : 1000

## Image Augmentation
  1. Train
      * CutMix
      * Resize (256)
      * RandomHorizontalFlip(0.5)
      * RandomCrop (224)
      * Normalize
  3. Validation
      * Resize (224)
      * Normalize

## Training 
  1. Optimizer : Adam
      * beta1=0.9, beta2=0.99
      * weight decay = 1e-4
  3. Scheudler : linear warmup and decay
      * warmup steps = 2000
  4. Epochs : 100
  5. Batch size : 256

## Data 
  1. ImageNet

## Reference
  1. VIT : https://arxiv.org/pdf/2010.11929.pdf
  2. CutMix : https://arxiv.org/pdf/1905.04899.pdf

