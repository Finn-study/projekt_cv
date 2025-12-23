import random
from pathlib import Path
import cv2
import shutil
import albumentations as A

# Pfade
INPUT_IMG_DIR = Path("data/split/images/test")
INPUT_LBL_DIR = Path("data/split/labels/test")

# OUTPUT-Pfade
OUT_IMG_DIR = Path("data/split/images/test_adverse")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR = Path("data/split/labels/test_adverse")
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# Augmentierungs-Pipelines
# 1. Nacht
t_night = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.19, -0.1),
        contrast_limit=(-0.05, 0.1),
        p=1.0),
    A.RandomGamma(
        gamma_limit=(85, 110),
        p=0.5),
    A.ISONoise(
        color_shift=(0.01, 0.03),
        intensity=(0.05, 0.2),
        p=0.3
    ),
    A.MotionBlur(blur_limit=2, p=0.2),
])

# 2. Regen
t_rain = A.Compose([
    A.RandomRain(
        brightness_coefficient=0.95,
        drop_length=10,
        drop_width=1,
        blur_value=3,
        p=1.0),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.05, 0.05),
        contrast_limit=(-0.1, 0.),
        p=0.7),
])

# 3. Gegenlicht
t_glare = A.Compose([
    A.RandomSunFlare(
        flare_roi=(0.0, 0.0, 1.0, 0.5),
        num_flare_circles_lower=1,
        num_flare_circles_upper=2,
        src_radius=20,
        src_intensity=0.1,
        p=0.7
    ),
    A.RandomBrightnessContrast(
        brightness_limit=(0.1, 0.2),
        contrast_limit=(-0.05, 0.15),
        p=1
    ),
    A.RandomGamma(
        gamma_limit=(105, 125),
        p=0.7),
])

def save_aug(img_path: Path, aug, suffix: str):
    """Speichert ein augmentiertes Bild und kopiert das zugehörige Label."""
    img = cv2.imread(str(img_path))
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = aug(image=img)["image"]
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    out_name = img_path.stem + f"_{suffix}" + img_path.suffix
    cv2.imwrite(str(OUT_IMG_DIR / out_name), out)

    lbl_src = INPUT_LBL_DIR / (img_path.stem + ".txt")
    if lbl_src.exists():
        shutil.copy(lbl_src, OUT_LBL_DIR / (img_path.stem + f"_{suffix}.txt"))


def main():
    # Liste aller Transformationen mit ihren Suffixen
    transforms = [
        (t_night, "night"),
        (t_rain, "rain"),
        (t_glare, "glare")
    ]

    imgs = [p for p in INPUT_IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    for i, p in enumerate(imgs):
        # Wähle für JEDES Originalbild ZUFÄLLIG EINE der drei Transformationen aus
        chosen_aug, suffix = random.choice(transforms)

        # Wende die ausgewählte Transformation an und speichere das Ergebnis
        save_aug(p, chosen_aug, suffix)

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(imgs)} Bilder verarbeitet…")

    print("Augmentierung abgeschlossen!")
    print(f"Es wurden {len(imgs)} neue Bilder im Ordner 'train_adverse' erstellt.")


if __name__ == "__main__":
    main()
