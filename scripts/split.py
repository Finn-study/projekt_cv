import random, math, shutil
from pathlib import Path
from collections import Counter, defaultdict

# ======= Pfade anpassen =======
ROOT = Path("data/processed")
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
OUT_DIR    = ROOT / "split"

# ======= Parameter =======
SPLIT = (0.80, 0.10, 0.10)    # train, val, test
SEED  = 0
DO_MOVE = False               # True = verschieben; False = kopieren
NEG_PER_POS = 1.0             # Deckel für Negative relativ zur Anzahl POSITIVER Bilder
CLASSES = {0:"stop", 1:"give_way", 2:"no_entry", 3:"priority_road", 4:"keep_right", 5:"go_straight"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".ppm"}

random.seed(SEED)

def find_image(stem: str):
    for ext in IMAGE_EXTS:
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def read_label_ids(stem: str):
    path = LABELS_DIR / f"{stem}.txt"
    ids=[]
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            try:
                ids.append(int(float(s.split()[0])))
            except Exception:
                pass
    return ids

def ensure_dirs():
    for split in ["train","val","test"]:
        (OUT_DIR/"images"/split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR/"labels"/split).mkdir(parents=True, exist_ok=True)

def copy_or_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if DO_MOVE:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

# 1) Inventur der Daten
stems = sorted({
    p.stem for p in IMAGES_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTS and "_aug_" not in p.stem.lower()
})

img_labels = {}
pos_imgs, neg_imgs = [], []
for s in stems:
    ids = [cid for cid in read_label_ids(s) if cid in CLASSES]
    lblset = set(ids)
    img_labels[s] = lblset
    if len(lblset) == 0:
        neg_imgs.append(s)
    else:
        pos_imgs.append(s)

# 2) Hintergrundbilder deckeln
max_neg = math.floor(NEG_PER_POS * len(pos_imgs))
if max_neg < len(neg_imgs):
    random.shuffle(neg_imgs)
    use_negs = neg_imgs[:max_neg]
else:
    use_negs = neg_imgs[:]  # alle erlaubt

# 3) Zielverteilungen berechnen
selected = pos_imgs + use_negs
random.shuffle(selected)

total_class_imgs = Counter()
for s in selected:
    for c in img_labels[s]:
        total_class_imgs[c] += 1

targets = {
    split: {c: round(total_class_imgs[c]*ratio) for c in CLASSES}
    for split, ratio in zip(["train","val","test"], SPLIT)
}

total_target_imgs = {
    split: round(len(selected)*ratio) for split, ratio in zip(["train","val","test"], SPLIT)
}

# 4) Stratifizierter Split
selected.sort(key=lambda s: (len(img_labels[s]), s), reverse=True)

assign = {}
cur_class_imgs = {split: Counter() for split in ["train","val","test"]}
cur_total = {split: 0 for split in ["train","val","test"]}

def split_score(split: str, labels: set):
    score = 0.0
    # Klassenabstand gewichten
    for c in CLASSES:
        want = targets[split][c]
        have = cur_class_imgs[split][c] + (1 if c in labels else 0)
        if want > 0:
            score += max(0, have - want) / want
        else:
            score += (1 if c in labels else 0) * 1.0
    tot_want = total_target_imgs[split]
    tot_have = cur_total[split] + 1
    if tot_want > 0:
        score += max(0, tot_have - tot_want) / tot_want
    return score

for s in selected:
    labels = img_labels[s]
    cand = []
    for split in ["train","val","test"]:
        cand.append((split_score(split, labels), cur_total[split], split))
    cand.sort()
    chosen = cand[0][2]

    assign[s] = chosen
    cur_total[chosen] += 1
    for c in labels:
        cur_class_imgs[chosen][c] += 1

# 5) Ausführung
ensure_dirs()
moved = 0
for s, split in assign.items():
    img = find_image(s)
    lbl = LABELS_DIR / f"{s}.txt"
    if img is None: continue

    copy_or_move(img, OUT_DIR/"images"/split/img.name)
    if lbl.exists():
        copy_or_move(lbl, OUT_DIR/"labels"/split/lbl.name)
    else:
        (OUT_DIR/"labels"/split/f"{s}.txt").write_text("", encoding="utf-8")
    moved += 1

print("\nSplit abgeschlossen. Dateien liegen unter:", OUT_DIR)

# 6) Reporting
def report(split):
    img_dir = OUT_DIR/"images"/split
    stems_split = [p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    neg = 0
    class_img = Counter()
    obj = Counter()
    for s in stems_split:
        ids = [cid for cid in read_label_ids(s) if cid in CLASSES]
        if not ids:
            neg += 1
        for c in set(ids):
            class_img[c] += 1
        for c in ids:
            obj[c] += 1
    print(f"\n[{split.upper()}] Bilder: {len(stems_split)} | Negative: {neg}")
    for c in sorted(CLASSES):
        print(f"  {c} ({CLASSES[c]}): {class_img[c]} Bilder, {obj[c]} Objekte")

for split in ["train","val","test"]:
    report(split)
