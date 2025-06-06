import os
import numpy as np
import logging
import pickle
import json
import shutil
import datetime

import matplotlib.pyplot as plt
from typing import Any, List

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import cv2
from PIL import Image


def cosine_similarity(t1, t2):
    return torch.dot(t1, t2) / (t1.norm() * t2.norm())

def read_meta_data(path: str):
    if not os.path.exists(path):
        meta_data = {
            "ids":[],
            "names":[],
            }
        with open(path, 'w') as f:
            json.dump(meta_data, f, indent=4)
        return meta_data
    with open(path, 'r') as f:
        return json.load(f)

class FaceDetection:
    def __init__(self, database_path: str = 'face_db.pkl',
                 log_level: int = logging.INFO, log_to_console: bool = True):
        self.mtcnn = MTCNN(image_size=160, margin=20)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.DATABASE_FOLDER = 'database'
        self.LOG_FOLDER = 'logs'
        self.SAMPLE_FOLDER = 'sample'

        os.makedirs(os.path.join(self.DATABASE_FOLDER, "Images"), exist_ok=True)
        os.makedirs(self.LOG_FOLDER, exist_ok=True)
        os.makedirs(self.SAMPLE_FOLDER, exist_ok=True)

        
        self.DB_PATH = os.path.join(self.DATABASE_FOLDER, database_path)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.LOG_FOLDER, f"model_log_{now}.log")

        handlers = [logging.FileHandler(log_file, mode='w')]
        if log_to_console:
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=handlers
        )
        self.logger = logging.getLogger("FaceDetection")

    def __load_database(self):
        try:
            if os.path.exists(self.DB_PATH):
                with open(self.DB_PATH, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
        return {}

    def __save_database(self, db, meta_data: json, source: str):
        try:
            with open(self.DB_PATH, 'wb') as f:
                pickle.dump(db, f)
            with open(os.path.join(self.DATABASE_FOLDER, 'meta_data.json'), 'w') as f:
                json.dump(meta_data, f, indent=4)
            shutil.copy(source, os.path.join(self.DATABASE_FOLDER, "Images", f"{meta_data['ids'][-1]}_{meta_data['names'][-1]}.png"))
        except Exception as e:
            self.logger.error(f"Failed to save database or copy image: {e}")

    def __get_embedding(self, source: str | np.ndarray | Any):
        try:
            if isinstance(source, str):
                img = Image.open(source).convert('RGB')
            elif isinstance(source, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Unsupported image type: must be file path or OpenCV image.")

            face = self.mtcnn(img)
            if face is None:
                return None
            return self.resnet(face.unsqueeze(0)).squeeze(0)
        except Exception as e:
            self.logger.error(f"Failed to extract embedding: {e}")
            return None

    def __find_most_similar(self, embedding, db):
        try:
            best_name, highest_sim = None, -1
            for name, db_embedding in db.items():
                sim = cosine_similarity(embedding, db_embedding).item()
                if sim > highest_sim:
                    best_name, highest_sim = name, sim
            return best_name, highest_sim
        except Exception as e:
            self.logger.error(f"Error while comparing embeddings: {e}")
            return None, -1

    def __get_embedding_and_best_match(self, image_path):
        embedding = self.__get_embedding(image_path)
        if embedding is None:
            self.logger.warning("No face detected")
            return None, None, None
        db = self.__load_database()
        if not db:
            return embedding, None, -1
        best_name, highest_sim = self.__find_most_similar(embedding, db)
        return embedding, best_name, highest_sim

    def register_face(self, name: str, image_path: str, threshold:float=0.8, user_id: str|int = 'auto'):
        try:
            meta_data_path = os.path.join(self.DATABASE_FOLDER, 'meta_data.json')
            meta_data = read_meta_data(meta_data_path)
            db = self.__load_database()

            embedding, matched_name, similarity = self.__get_embedding_and_best_match(image_path)
            if embedding is None:
                return

            if similarity >= threshold:
                self.logger.warning(f"Similar face found: {matched_name} (similarity = {similarity:.4f}) — Overwriting")
                db[matched_name] = embedding
            else:
                if user_id == 'auto':
                    user_id = max(meta_data['ids'], default=0) + 1
                meta_data['ids'].append(user_id)
                meta_data['names'].append(name)
                db[name] = embedding
                self.logger.info(f"Successfully registered new user: {name}")

            img = Image.open(image_path).convert('RGB')
            box, _ = self.mtcnn.detect(img)

            if box is not None:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                x1, y1, x2, y2 = [int(b) for b in box[0]]
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

                plt.figure(figsize=(6, 6))
                plt.axis('off')
                plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                save_path = os.path.join(self.SAMPLE_FOLDER, f"bbox_{meta_data['ids'][-1]}_{meta_data['names'][-1]}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            self.__save_database(db, meta_data, image_path)
        except Exception as e:
            self.logger.error(f"Failed to register face for '{name}': {e}")

    def verify_face(self, source: str | np.ndarray | Any, threshold=0.6):
        try:
            db = self.__load_database()
            if not db:
                self.logger.error("No registered faces found")
                return None

            embedding = self.__get_embedding(source)
            if embedding is None:
                self.logger.warning("No face detected")
                return None

            best_match, similarity = self.__find_most_similar(embedding, db)

            if similarity > threshold:
                meta_data_path = os.path.join(self.DATABASE_FOLDER, 'meta_data.json')
                meta_data = read_meta_data(meta_data_path)
                if best_match in meta_data["names"]:
                    idx = meta_data["names"].index(best_match)
                    user_id = meta_data["ids"][idx]
                    self.logger.info(f"Matched with: {best_match} (ID = {user_id}, similarity = {similarity:.4f})")
                    return {"id": user_id, "name": best_match, "similarity": similarity}
                else:
                    self.logger.info(f"Matched with: {best_match} (similarity = {similarity:.4f}) — ID not found")
                    return {"id": None, "name": best_match, "similarity": similarity}
            else:
                self.logger.info(f"No match found (max similarity = {similarity:.4f})")
                return None
        except Exception as e:
            self.logger.error(f"Error during face verification: {e}")
            return None

    def rename_user(self, old_name: str, new_name: str):
        try:
            meta_data_path = os.path.join(self.DATABASE_FOLDER, 'meta_data.json')
            meta_data = read_meta_data(meta_data_path)
            db = self.__load_database()

            if old_name not in db or old_name not in meta_data["names"]:
                self.logger.warning(f"Cannot rename: '{old_name}' not found.")
                return

            db[new_name] = db.pop(old_name)

            idx = meta_data["names"].index(old_name)
            user_id = meta_data["ids"][idx]
            meta_data["names"][idx] = new_name

            old_image = os.path.join(self.DATABASE_FOLDER, "Images", f"{user_id}_{old_name}.png")
            new_image = os.path.join(self.DATABASE_FOLDER, "Images", f"{user_id}_{new_name}.png")
            if os.path.exists(old_image):
                os.rename(old_image, new_image)

            old_bbox = os.path.join(self.SAMPLE_FOLDER, f"bbox_{user_id}_{old_name}.png")
            new_bbox = os.path.join(self.SAMPLE_FOLDER, f"bbox_{user_id}_{new_name}.png")
            if os.path.exists(old_bbox):
                os.rename(old_bbox, new_bbox)

            with open(self.DB_PATH, 'wb') as f:
                pickle.dump(db, f)
            with open(meta_data_path, 'w') as f:
                json.dump(meta_data, f, indent=4)

            self.logger.info(f"Renamed '{old_name}' ➝ '{new_name}'")
        except Exception as e:
            self.logger.error(f"Failed to rename user '{old_name}' to '{new_name}': {e}")