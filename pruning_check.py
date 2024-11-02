import json
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, 'r') as f:
        data = json.load(f)

    import pdb
    pdb.set_trace()
