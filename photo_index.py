import json
import pandas as pd
from pathlib import Path

PHOTO_JSON = Path("data/raw_photos/photos.json")
PHOTO_DIR  = Path("data/raw_photos/photos")

# Priority order for photo labels
LABEL_PRIORITY = {"food": 0, "inside": 1, "drink": 2, "outside": 3, "menu": 4}

records = []
with open(PHOTO_JSON, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        img_path = PHOTO_DIR / f"{r['photo_id']}.jpg"
        if img_path.exists():
            records.append({
                "business_id": r["business_id"],
                "photo_id":    r["photo_id"],
                "label":       r["label"],
                "path":        str(img_path),
                "priority":    LABEL_PRIORITY.get(r["label"], 99),
            })

df = pd.DataFrame(records)

# Keep only best photo per business (lowest priority number = best)
best = (
    df.sort_values("priority")
      .groupby("business_id")
      .first()
      .reset_index()[["business_id", "photo_id", "label", "path"]]
)

print(f"Total usable photos: {len(df):,}")
print(f"Unique businesses with photos: {len(best):,}")
print(f"Label distribution of best photos:")
print(best["label"].value_counts())

best.to_parquet("data/processed/photo_index.parquet", index=False)
print(f"\nSaved -> data/processed/photo_index.parquet")
