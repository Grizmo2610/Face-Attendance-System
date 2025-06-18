import os
import shutil
import argparse
import logging
import pandas as pd
from model import FaceDetection

import os
import logging
import pandas as pd
from model import FaceDetection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def batch_register_faces(csv_path: str):
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return

    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        return

    required_columns = {'name', 'path'}
    if not required_columns.issubset(set(data.columns)):
        logging.error(f"CSV must contain columns: {required_columns}")
        return

    model = FaceDetection()

    for idx, row in data.iterrows():
        name = row['name']
        source = row['path']

        if not os.path.exists(source):
            logging.warning(f"Image not found: {source}")
            continue

        try:
            model.register_face(name, source)
            logging.info(f"Registered: {name} from {source}")
        except Exception as e:
            logging.error(f"Failed to register {name} from {source}: {e}")

