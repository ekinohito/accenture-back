import albumentations as A
from albumentations.pytorch import ToTensorV2


transform_test = A.Compose([
    A.Resize(width=224, height=224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
    
])