import json

def load_hypergraphs(file_path="data/sample_hypergraph.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
