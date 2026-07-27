import json
from pathlib import Path

# Path to: src/scraping/new_scraped_data_ml_project
json_folder = Path(__file__).resolve().parents[1] / "scraping" / "new_scraped_data_ml_project"

# Load all JSON files
with open(json_folder / "patent_0_500.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

with open(json_folder / "patents_502-1500.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)


with open(json_folder / "patent_2503_3503.json", "r", encoding="utf-8") as f3:
    data3 = json.load(f3)

with open(json_folder / "patent_3503_4503.json", "r", encoding="utf-8") as f4:
    data4 = json.load(f4)

with open(json_folder / "patent_4503_5503.json", "r", encoding="utf-8") as f5:
    data5 = json.load(f5)

with open(json_folder / "patent_5503_6000.json", "r", encoding="utf-8") as f6:
    data6 = json.load(f6)

with open(json_folder / "patent_6000_7000_batch1.json", "r", encoding="utf-8") as f7:
    data7 = json.load(f7)

with open(json_folder / "patent_7000-8000.json", "r", encoding="utf-8") as f8:
    data8 = json.load(f8)

with open(json_folder / "patent_8000_9000.json", "r", encoding="utf-8") as f9:
    data9 = json.load(f9)

with open(json_folder / "patent_9000_10000.json", "r", encoding="utf-8") as f10:
    data10 = json.load(f10)

with open(json_folder / "patent_10000_11000.json", "r", encoding="utf-8") as f11:
    data11 = json.load(f11)

with open(json_folder / "patent_11000_12000.json", "r", encoding="utf-8") as f12:
    data12 = json.load(f12)

with open(json_folder / "patent_new_12500_13500.json", "r", encoding="utf-8") as f13:
     data13 = json.load(f13)

with open(json_folder / "patent_new_13500_14500.json", "r", encoding="utf-8") as f14:
     data14 = json.load(f14)

with open(json_folder / "patent_new_14500_15500.json", "r", encoding="utf-8") as f15:
     data15 = json.load(f15)

with open(json_folder / "patent_new_15500_16500.json", "r", encoding="utf-8") as f16:
     data16 = json.load(f16)

with open(json_folder / "patent_new_16500_17500.json", "r", encoding="utf-8") as f17:
     data17 = json.load(f17)

with open(json_folder / "patent_new_17500_18500.json", "r", encoding="utf-8") as f18:
     data18 = json.load(f18)

with open(json_folder / "patent_new_18500_19500.json", "r", encoding="utf-8") as f19:
     data19 = json.load(f19)

with open(json_folder / "patent_new_19500_20500.json", "r", encoding="utf-8") as f20:
     data20 = json.load(f20)

# Merge using + operator
merged_data = (
    data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8 +
    data9 + data10 + data11 + data12 
     + data13 + data14 + data15 +
     data16 + data17 + data18 + data19 + data20
)

# Save merged file in src/preprocessing
output_file = Path(__file__).resolve().parent / "merged.json"

with open(output_file, "w", encoding="utf-8") as output_file:
    json.dump(merged_data, output_file, indent=4, ensure_ascii=False)

print(f"Merged {len(merged_data)} patents successfully into {output_file}")