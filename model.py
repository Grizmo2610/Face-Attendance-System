import os
import shutil
import logging
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import torch

from facenet_pytorch import MTCNN, InceptionResnetV1

from logger import MyLogger
from Database import UserDatabase
from util import *

logger = MyLogger()

class FaceDetection:
    def __init__(self, database_path: str = 'face_db.pt',
                 log_level: int = logging.INFO, log_to_console: bool = True):
        self.mtcnn = MTCNN(image_size=160, margin=20, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            
        self.DATABASE_FOLDER = 'database'
        self.LOG_FOLDER = 'logs'
        self.SAMPLE_FOLDER = 'sample'
        self.database_path = database_path
        
        os.makedirs(os.path.join(self.DATABASE_FOLDER, "Images"), exist_ok=True)
        os.makedirs(self.LOG_FOLDER, exist_ok=True)
        os.makedirs(self.SAMPLE_FOLDER, exist_ok=True)

        self.__database = UserDatabase(database_path)
        self.__db = self.__database.get_db()
        
        self.logger = logger
        self.logger.setup(log_level, log_to_console, self.LOG_FOLDER)
        
        if len(self.__db) != len(self.__database.meta_data_query("ids")):
            self.logger.warning("DB and meta_data out of sync!")

    def detect_face(self, img):
        face = self.mtcnn(img)
        if face is None:
            return None
        return face

    def get_bbox(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        box, _ = self.mtcnn.detect(img_pil)
        if box is not None and box[0] is not None:
            return box
        return None
    
    def __get_embedding(self, source: str | np.ndarray | Image.Image):
        try:
            if isinstance(source, str):
                img = Image.open(source).convert('RGB')
            elif isinstance(source, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            elif isinstance(source, Image.Image):
                img = source.convert("RGB")
            else:
                raise ValueError("Unsupported image type: must be file path or OpenCV image.")

            face = self.detect_face(img)
            return self.resnet(face.unsqueeze(0)).squeeze(0)
        except Exception as e:
            self.logger.error(f"Failed to extract embedding: {e}")
            return None

    def __find_most_similar(self, embedding):
        try:
            user_id, highest_sim = None, -1
            for id, db_embedding in self.__db.items():
                sim = cosine_similarity(embedding, db_embedding).item()
                if sim > highest_sim:
                    user_id, highest_sim = id, sim
            return user_id, highest_sim
        except Exception as e:
            self.logger.error(f"Error while comparing embeddings: {e}")
            return None, -1

    def __get_embedding_and_best_match(self, source: str| np.ndarray):
        embedding = self.__get_embedding(source)
        if embedding is None:
            self.logger.warning("No face detected")
            return None, None, None
        if not self.__db:
            return embedding, None, -1
        user_id, highest_sim = self.__find_most_similar(embedding)
        
        return embedding, user_id, highest_sim

    def findNameById(self, user_id):
        if user_id in self.__database.meta_data_query("ids"):
            idx = self.__database.meta_data_query("ids").index(user_id)
            return self.__database.meta_data_query("names")[idx]
        return None
    
    def findIdByName(self, name):
        return [self.__database.meta_data_query('ids')[i] for i in range(len(self.__database.meta_data_query('names'))) if self.__database.meta_data_query('names')[i] == name]

    def register_face(self, name: str, source: str | np.ndarray, threshold:float=0.8):
        try:
            embedding, matched_id, similarity = self.__get_embedding_and_best_match(source)
            matched_name = self.findNameById(matched_id) if matched_id is not None else "Unknown"
            
            if isinstance(source, str):
                img = Image.open(source).convert('RGB')
            elif isinstance(source, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Unsupported image type: must be file path or OpenCV image.")
            
            if embedding is None:
                raise ValueError("Image is none!")

            if similarity >= threshold:
                self.__db[matched_id] = embedding
                self.__database.save_database(img)
                return self.logger.warning(f"Similar face found: {matched_name} (similarity = {similarity * 100:.2f}%) — Overwriting"), "blue"
            else:
                user_id = generate_unique_id(self.__database.meta_data_query("ids"))   
                self.__database.update_meta_data("ids", user_id)
                self.__database.update_meta_data("names", name)
                self.__db[user_id] = embedding
                
                self.__database.save_database(img)
                return self.logger.info(f"Successfully registered new user: {safe_filename(name)} | ID: {user_id} "), "green"
            # box, _ = self.mtcnn.detect(img)

            # if box is not None and box[0] is not None:
            #     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            #     x1, y1, x2, y2 = [int(b) for b in box[0]]
            #     cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #     plt.figure(figsize=(6, 6))
            #     plt.axis('off')
            #     plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            #     filename = safe_filename(f"bbox_{self.__database.meta_data_query('ids')[-1]}_{self.__database.meta_data_query('ids')[-1]}.png")
            #     save_path = os.path.join(self.SAMPLE_FOLDER, filename)
            #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            #     plt.close('all')

        except Exception as e:
            return self.logger.error(f"Failed to register face for '{safe_filename(name)}': {e}"), "red"

    def verify_face(self, source: str | np.ndarray | Any, threshold: float=0.6):
        try:
            embedding, best_match, similarity = self.__get_embedding_and_best_match(source)
            if embedding is None:
                self.logger.warning("No face detected")
                return None

            if similarity >= threshold:
                if best_match in self.__database.meta_data_query("ids"):
                    idx = self.__database.meta_data_query("ids").index(best_match)
                    name = self.__database.meta_data_query("names")[idx]
                    user_id = best_match
                    self.logger.info(f"Matched with: {name} (ID = {user_id}, similarity = {similarity:.4f})")
                    return {"id": user_id, "name": name, "similarity": similarity}
                else:
                    self.logger.info(f"Matched with: {best_match} (similarity = {similarity:.4f}) — ID not found")
                    return {"id": None, "name": best_match, "similarity": similarity}
            else:
                self.logger.info(f"No match found (max similarity = {similarity:.4f})")
                return {"id": None, "name": None, "similarity": -1.0}
        except Exception as e:
            self.logger.error(f"Error during face verification: {e}")
            return {'id': None, "name": None, "similarity": -1.0}

    def rename_user(self, user_id: str, new_name: str):
        if user_id not in self.__database.meta_data_query("ids"):
            self.logger.warning(f"User ID {user_id} cannot be founded in database! nothing changed")
            return
        old_name = self.__database.rename_user_by_id(user_id, new_name)
        try:
            if old_name:
                self.__rename_file(old_name, new_name, user_id)
                self.__database.save_database()
            self.logger.info(f"Renamed '{old_name}' -> '{new_name}'")
        except Exception as e:
            self.logger.error(f"Failed to rename user '{old_name}' to '{new_name}': {e}")

    def __rename_file(self, old_name, new_name, user_id):
        old_image = os.path.join(self.DATABASE_FOLDER, "Images", f"{user_id}_{old_name}.png")
        new_image = os.path.join(self.DATABASE_FOLDER, "Images", f"{user_id}_{new_name}.png")
        if os.path.exists(old_image):
            os.rename(old_image, new_image)

        old_bbox = os.path.join(self.SAMPLE_FOLDER, f"bbox_{user_id}_{old_name}.png")
        new_bbox = os.path.join(self.SAMPLE_FOLDER, f"bbox_{user_id}_{new_name}.png")
        if os.path.exists(old_bbox):
            os.rename(old_bbox, new_bbox)
            
    def reset_system(self):
        try:
            for folder in [self.DATABASE_FOLDER, self.SAMPLE_FOLDER]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    self.logger.info(f"Deleted folder: {folder}")
            
            os.makedirs(os.path.join(self.DATABASE_FOLDER, "Images"), exist_ok=True)
            os.makedirs(self.SAMPLE_FOLDER, exist_ok=True)
            self.__database = UserDatabase(self.database_path)
            self.__db = self.__database.get_db()
            
            init_metadata(self.__database.meta_data_path)
            self.logger.info("System reset successfully: All data removed and structure re-initialized.")
        except Exception as e:
            print(f"[ERROR] Failed to reset system: {e}")
            
    def delete_user(self, user_id: int):
        if user_id not in self.__database.meta_data_query("ids"):
            self.logger.warning(f"User ID {user_id} not found.")
            return
        idx = self.__database.meta_data_query("ids").index(user_id)
        name = self.__database.meta_data_query("names")[idx]

        del self.__database.meta_data_query("ids")[idx]
        del self.__database.meta_data_query("names")[idx]
        self.__db.pop(user_id, None)

        self.__delete_files(user_id, name)
        self.__database.save_database()
        self.logger.info(f"Deleted user {user_id} ({name})")

    def __delete_files(self, user_id, name):
        files = [
            os.path.join(self.DATABASE_FOLDER, "Images", f"{user_id}_{name}.png"),
            os.path.join(self.SAMPLE_FOLDER, f"bbox_{user_id}_{name}.png")
        ]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
                
    def get_all_users(self):
        return [{"id": id, "name": name} for name, id in zip(self.__database.meta_data_query("names"), self.__database.meta_data_query("ids"))]