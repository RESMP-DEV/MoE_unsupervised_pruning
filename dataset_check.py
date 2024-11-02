import json
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, 'r') as f:
        data = json.load(f)

    sample_type_map = {}
    for basic_data in data:
        sample_type = basic_data["source"].split("/")[1]
        if sample_type not in sample_type_map:
            sample_type_map[sample_type] = 0
        sample_type_map[sample_type] += 1

    for k, v in sample_type_map.items():
        print(k, v)
