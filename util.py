import os
import re
import json
import unicodedata
from typing import Any

import torch

def cosine_similarity(t1, t2):
    if t1.norm() == 0 or t2.norm() == 0:
        return torch.tensor(0.0)
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()


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