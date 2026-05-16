import json
import random

data = json.load(open(r"c:\D_Folder\python_projects\Defect severity prediction\data\image_attributes.json"))
good = [d for d in data if "error" not in d.get("attributes", {})]

random.seed(42)
samples = random.sample(good, 3)

for s in samples:
    label = s["severity_label"]
    fname = s["image_path"].split("\\")[-1]
    attrs = json.dumps(s["attributes"], indent=2)
    print(f"[{label}] {fname}")
    print(attrs)
    print()
