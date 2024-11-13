from albumentations.pytorch import ToTensorV2
import albumentations as album

def get_train_preprocessing():
    transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        album.RandomRotate90(p=0.5),
        album.RandomBrightnessContrast(),
        album.OneOf([
            album.GaussNoise(),
            album.MotionBlur(p=0.2),
            album.MedianBlur(blur_limit=3, p=0.1),
            album.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        album.OneOf([
            album.CLAHE(clip_limit=2),
        ], p=0.3),
        album.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ]
    return album.Compose(transform)

def get_valid_preprocessing():
    transform = [
        album.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ]
    return album.Compose(transform)

def get_test_preprocessing():
    transform = [
        album.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ]
    return album.Compose(transform)
