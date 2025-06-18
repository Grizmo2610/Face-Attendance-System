import os
import re
import json
import unicodedata
import uuid
from typing import Any
import torch.nn.functional as F

import torch

def generate_unique_id(existing_ids):
    while True:
        new_id = uuid.uuid4().hex[:8]
        if new_id not in existing_ids:
            return new_id
        
def cosine_similarity(t1, t2):
    denom = t1.norm() * t2.norm()
    if denom == 0:
        return torch.tensor(0.0)
    return torch.dot(t1, t2) / denom

def init_metadata(path: str):
    meta_data = {
        "ids":[],
        "names":[],
        }
    with open(path, 'w') as f:
        json.dump(meta_data, f, indent=4)
    return meta_data

def read_meta_data(path: str):
    if not os.path.exists(path):
        return init_metadata(path)
    with open(path, 'r') as f:
        return json.load(f)

def safe_filename(name):
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^\w\-]', '_', name).lower()