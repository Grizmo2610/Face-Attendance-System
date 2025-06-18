import os
import json
import shutil
from typing import Any

import numpy as np

import cv2
from PIL import Image

from logger import MyLogger
from util import *

logger = MyLogger()

class UserDatabase:
    def __init__(self, database_path: str):
        self.DATABASE_FOLDER = 'database'
        self.DB_PATH = os.path.join(self.DATABASE_FOLDER, database_path)
        self.meta_data_path = os.path.join(self.DATABASE_FOLDER, 'meta_data.json')
        
        self.__meta_data = read_meta_data(self.meta_data_path)
        self.__db = self.__load_database()
        
        self.logger = logger
    def __load_database(self):
        if os.path.exists(self.DB_PATH):
            try:
                with open(self.DB_PATH, 'rb') as f:
                    return torch.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load database: {e}")
        return {}
    
    def meta_data_query(self, key: str):
        return self.__meta_data[key]
        
    def meta_data_structure(self, d, indent=0):
        for key, value in d.items():
            prefix = "  " * indent + f"{key}: "
            if isinstance(value, dict):
                print(prefix)
                self.meta_data_structure(value, indent + 1)
            elif isinstance(value, list):
                if value:
                    element_type = type(value[0]).__name__
                else:
                    element_type = "unknown"
                print(f"{prefix}list[{element_type}]")
            else:
                print(f"{prefix}{type(value).__name__}")
        
    def __save_image(self, source: str| np.ndarray| Image.Image = None):
        save_path = os.path.join(self.DATABASE_FOLDER, "Images", f"Default.png")
        if self.__meta_data["ids"] and self.__meta_data["names"]:
            filename = safe_filename(f"{self.__meta_data['ids'][-1]}_{self.__meta_data['names'][-1]}") + '.png'
            save_path = os.path.join(self.DATABASE_FOLDER, "Images", filename)

        if isinstance(source, str):
            shutil.copy(source, save_path)
        elif isinstance(source, np.ndarray):
            cv2.imwrite(save_path, source)
        elif isinstance(source, Image.Image):
            source.save(save_path)
            
        elif source is None:
            self.logger.warning("Database saved without new image!")
    
    def save_database(self, source: str| np.ndarray| Image.Image = None):
        try:
            torch.save(self.__db, self.DB_PATH)
            with open(self.meta_data_path, 'w') as f:
                json.dump(self.__meta_data, f, indent=4)
            self.__save_image(source)
        except Exception as e:
            self.logger.error(f"Failed to save database or copy image: {e}")
    
    def update_meta_data(self, key: str, value):
        try:
            self.__meta_data[key].append(value)
        except KeyError as e:
            self.logger.error(e)
            self.logger.error(f"Key {key} is not a valid key in meta data!")
            
    def set_name_by_index(self, index: int, new_name: str):
        try:
            self.__meta_data["names"][index] = new_name
        except IndexError:
            self.logger.error(f"Index {index} out of range while renaming.")

    def rename_user_by_id(self, user_id: int, new_name: str):
        try:
            index = self.__meta_data["ids"].index(user_id)
            old_name = self.__meta_data["names"][index]
            self.__meta_data["names"][index] = new_name
            return old_name
        except ValueError:
            self.logger.error(f"User ID {user_id} not found.")
            return None
        
    def get_db(self):
        return self.__db