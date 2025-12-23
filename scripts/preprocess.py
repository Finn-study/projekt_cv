import os
import sys
from PIL import Image
from collections import defaultdict, Counter

# Pfade
ROOT = "data/raw_gtsdb"
OUT  = "data/processed"
INCLUDE_NEGATIVES = True

# Mapping: Original-ID -> neue YOLO-ID
keep_map = {
    14: 0,  # stop
    13: 1,  # give_way
    17: 2,  # no_entry
    12: 3,  # priority_road
    38: 4,  # keep_right
    35: 5,  # go_straight
}

# Annotations laden & filtern
gt_file = os.path.join(ROOT, "gt.txt")
if not os.path.isfile(gt_file):
    sys.exit(f"gt.txt nicht gefunden unter: {gt_file}")

ann = defaultdict(list)
cls_count = Counter()

with open(gt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        fn, xmin, ymin, xmax, ymax, cls = line.split(";")
        xmin, ymin, xmax, ymax, cls = map(int, [xmin, ymin, xmax, ymax, cls])
        if cls in keep_map:
            new_id = keep_map[cls]
            ann[fn].append((xmin, ymin, xmax, ymax, new_id))
            cls_count[new_id] += 1

# Ausgabestruktur
img_out = os.path.join(OUT, "images")
lab_out = os.path.join(OUT, "labels")
os.makedirs(img_out, exist_ok=True)
os.makedirs(lab_out, exist_ok=True)

# Konvertierung
imgs = [f for f in os.listdir(ROOT) if f.lower().endswith(".ppm")]
imgs.sort()

converted, with_labels, empties = 0, 0, 0

for fn in imgs:
    src = os.path.join(ROOT, fn)
    stem = os.path.splitext(fn)[0]

    # Bild nach JPG konvertieren
    im = Image.open(src).convert("RGB")
    W, H = im.size
    dst_img = os.path.join(img_out, stem + ".jpg")
    im.save(dst_img, quality=95)

    # Label schreiben
    lines = []
    for (xmin, ymin, xmax, ymax, new_cls) in ann.get(fn, []):
        xc = ((xmin + xmax) / 2.0) / W
        yc = ((ymin + ymax) / 2.0) / H
        w  = (xmax - xmin) / W
        h  = (ymax - ymin) / H
        # Clamp
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w  = min(max(w , 0.0), 1.0)
        h  = min(max(h , 0.0), 1.0)
        lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    label_path = os.path.join(lab_out, stem + ".txt")
    if lines:
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))
        with_labels += 1
    else:
        if INCLUDE_NEGATIVES:
            open(label_path, "w", encoding="utf-8").close()  # leere .txt
            empties += 1
        else:
            pass

    converted += 1

print(f"Konvertiert: {converted} Bilder")
print(f"mit Labels:  {with_labels}")
print(f"leere Labels (Negatives): {empties}")
print("Verteilung pro Klasse (neue IDs):", dict(cls_count))
