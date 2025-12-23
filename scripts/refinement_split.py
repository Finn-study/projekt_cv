import shutil, random
from pathlib import Path
from collections import Counter

# Pfade
ROOT = Path("data/processed/split")
IMG = {s: ROOT/"images"/s for s in ("train","val","test")}
LBL = {s: ROOT/"labels"/s for s in ("train","val","test")}
IMAGE_EXTS = {".jpg",".jpeg",".png",".ppm"}
SEED = 0
random.seed(SEED)

MIN_VAL  = {0:5, 2:5, 5:5}   # stop, no_entry, go_straight
MIN_TEST = {0:5, 2:5, 5:5}

TARGETS = {0:"stop", 2:"no_entry", 5:"go_straight"}

def read_items(lbl_path: Path):
    items=[]
    if not lbl_path.exists(): return items
    with lbl_path.open("r",encoding="utf-8") as f:
        for ln in f:
            s=ln.strip()
            if not s: continue
            parts=s.split()
            if len(parts)<5: continue
            try:
                c=int(float(parts[0])); x,y,w,h=map(float, parts[1:5])
            except: continue
            items.append((c,x,y,w,h))
    return items

def img_path_for(stem: str, folder: Path):
    for ext in IMAGE_EXTS:
        p = folder/f"{stem}{ext}"
        if p.exists(): return p
    return None

def get_stems(folder: Path):
    return sorted([p.stem for p in folder.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and "_aug_" not in p.stem.lower()])

def count_objects(split: str):
    obj = Counter(); neg=0
    for s in get_stems(IMG[split]):
        ids=[c for c, *_ in read_items(LBL[split]/f"{s}.txt")]
        if not ids: neg+=1
        for i in ids: obj[i]+=1
    return obj, neg

def pure_train_candidates(k: int):
    """Sucht Bilder in Trainingsdaten, die nur Klasse k enthalten."""
    out=[]
    for s in get_stems(IMG["train"]):
        items = read_items(LBL["train"]/f"{s}.txt")
        if not items: continue
        cls_set = set(c for c, *_ in items)
        if cls_set == {k}:
            n = sum(1 for c,*_ in items if c==k)
            out.append((s,n))
    out.sort(key=lambda t: t[1])
    return out

def move_one(stem: str, dst_split: str):
    src_img = img_path_for(stem, IMG["train"])
    src_lbl = LBL["train"]/f"{stem}.txt"
    if not src_img or not src_lbl.exists():
        return False
    dst_img = IMG[dst_split]/src_img.name
    dst_lbl = LBL[dst_split]/src_lbl.name
    shutil.move(str(src_img), str(dst_img))
    shutil.move(str(src_lbl), str(dst_lbl))
    return True

# Vorherige Verteilung
print("Status vor Feinjustierung")
for sp in ("train","val","test"):
    obj,neg = count_objects(sp)
    print(f"[{sp.upper()}] neg={neg}  " + "  ".join(f"{k}:{obj[k]}" for k in sorted(TARGETS)))

# Bedarf ermitteln
obj_val,  _ = count_objects("val")
obj_test, _ = count_objects("test")

needs_val  = {k: max(0, MIN_VAL[k]  - obj_val[k])  for k in TARGETS}
needs_test = {k: max(0, MIN_TEST[k] - obj_test[k]) for k in TARGETS}

# Kandidaten aus Train je Klasse
cands = {k: pure_train_candidates(k) for k in TARGETS}

# Verschieben: zuerst Val, dann Test
for split, needs in (("val", needs_val), ("test", needs_test)):
    for k in TARGETS:
        need = needs[k]
        if need <= 0: continue
        pool = cands[k]
        i = 0
        while need > 0 and i < len(pool):
            stem, n = pool[i]; i += 1
            if move_one(stem, split):
                need -= n
                # entferne aus anderen Pools, damit nicht doppelt verschoben wird
                for kk in cands:
                    cands[kk] = [t for t in cands[kk] if t[0] != stem]
        if need > 0:
            print(f"! WARNUNG: Nicht genug reine {TARGETS[k]}-Bilder für {split}. Fehlend ~{need} Objekte.")

# Neue Verteilung
for sp in ("train","val","test"):
    obj,neg = count_objects(sp)
    print(f"\n[{sp.upper()}] neg={neg}")
    for k in sorted(TARGETS):
        print(f"  {k} ({TARGETS[k]}): {obj[k]} Objekte")
