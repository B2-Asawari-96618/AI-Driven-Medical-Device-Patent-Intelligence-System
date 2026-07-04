
import json

# Load the first JSON file
with open("patents_502-1500.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

# Load the second JSON file
with open("patents_1503-2502.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# Merge the data
merged_data = data1 + data2

# Save the combined data
with open("merged.json", "w", encoding="utf-8") as output_file:
    json.dump(merged_data, output_file, indent=4, ensure_ascii=False)

print(f"Merged {len(merged_data)} patents successfully.")