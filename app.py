from model import FaceDetection
import logging
import time
import os
import json
import sys
import shutil


model = FaceDetection()

def reset():    
    for folder in [model.LOG_FOLDER, model.SAMPLE_FOLDER, model.DATABASE_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    

model.register_face("Hoang Tuan Tu","images/Image.png")

model.verify_face("images/Check.jpg")