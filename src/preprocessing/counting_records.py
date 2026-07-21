import json
import os

folder = "../scraping/scraped_data"

files = [
    "patent_0_500.json",
    "patents_502-1500.json",
    "patents_2503-6000.json",
    "patents_6000_12000.json",
    "patent_12000_20000.json"
]

total = 0

for file in files:
    path = os.path.join(folder, file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        count = len(data)
        total += count
        print(f"{file}: {count} records")

print(f"\nTotal records: {total}")