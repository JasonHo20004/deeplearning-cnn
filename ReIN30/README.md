# Animal Image Dataset

5-class animal image dataset for image classification.

## Structure

```
dataset/
├── train/
│   ├── dog/
│   ├── cat/
│   ├── bird/
│   ├── horse/
│   └── elephant/
├── test/
│   ├── dog/
│   ├── cat/
│   ├── bird/
│   ├── horse/
│   └── elephant/
└── README.md
```

## Collection

Images are downloaded from Bing Image Search via `icrawler`.  
Target: ~300–1000 images per class (80% train / 20% test).

To (re)download or refresh the dataset:

```bash
python scripts/download_animal_dataset.py
```

## Usage with PyTorch

```python
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = ImageFolder('dataset/train', transform=transform)
test_data = ImageFolder('dataset/test', transform=transform)
```
