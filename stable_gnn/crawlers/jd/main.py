import os
import logging
import json
from pathlib import Path
from arango import ArangoClient

from gns.dataset.dataset_folder import DATASET_FOLDER
from gns.dataset.tech_dataset import TechDataset

logger = logging.getLogger(__name__)


def get_json(name='jd_data2'):
    # Initial params
    file = os.path.join(DATASET_FOLDER, 'TechDataset') + '/' + name + '.json'

    # Check if file is exists
    file_path = Path(file)
    dataset = TechDataset(name)
    if not file_path.is_file():
        # Download JSON if not
        dataset.download()

    # Parse JSON
    f = open(file, "rb")
    data = json.loads(f.read())
    f.close()

    return data


# Get data
jsonObj = get_json()

# Extract data
nodes = jsonObj['nodes']
edges = jsonObj['links']

# Initialize the ArangoDB client.
client = ArangoClient()

# Connect to "_system" database as root user.
db = client.db('_system', username='root', password='root_pass')

# Create connection with nodes collections
if not db.has_collection('nodes'):
    nodesCollection = db.create_collection(name="nodes", key_generator='autoincrement')
else:
    nodesCollection = db.collection('nodes')

# Import nodes
for node in nodes:
    name = node['id']
    exists = nodesCollection.find(filters={"name": name})
    if int(exists.count()) < 1:
        print("inserting", {"name": node['id']})
        nodesCollection.insert({"name": node['id']})

# Create connection with edges collections
if not db.has_collection('edges'):
    edgesCollection = db.create_collection(name="edges", key_generator='autoincrement', edge=True)
else:
    edgesCollection = db.collection('edges')

# Import edges
for edge in edges:
    source = edge['source']
    target = edge['target']
    weight = edge['value']

    sourceId = nodesCollection.find(filters={"name": source}).next()['_id']
    targetId = nodesCollection.find(filters={"name": target}).next()['_id']

    exists = edgesCollection.find(filters={"_from": sourceId, "_to": targetId})
    if int(exists.count()) < 1:
        print("inserting", {"_from": sourceId, "_to": targetId})
        edgesCollection.insert({"_from": sourceId, "_to": targetId})
