import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
import re
from model import FaceDetection
import logging

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(255 * ((i / 255.0) ** invGamma)) for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def update_frame():
    global paused, last_frame
    if not paused:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

            gamma_value = gamma_slider.get() / 100
            gamma_value = max(gamma_value, 0.01)
            adjusted = adjust_gamma(resized, gamma=gamma_value)

            rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            last_frame = frame.copy()
    root.after(15, update_frame)

def toggle_pause(event=None):
    global paused
    paused = not paused
    if paused:
        confirm_frame.pack(pady=10)
    else:
        confirm_frame.pack_forget()
        name_frame.pack_forget()
        message_label.pack_forget()

def on_yes():
    confirm_frame.pack_forget()
    name_entry.delete(0, tk.END)
    name_frame.pack(pady=10)

def on_no():
    global paused
    paused = False
    confirm_frame.pack_forget()
    name_frame.pack_forget()
    message_label.pack_forget()

def save_image():
    global paused
    name = name_entry.get().strip()
    
    if not name or not re.match(r'^[\w\s-]+$', name):
        show_message("❌ Registration failed: Invalid name", "red")
        return

    try:
        os.makedirs("sample/registered", exist_ok=True)
        filename = f"sample/registered/{name}_{int(time.time())}.jpg"
        cv2.imwrite(filename, last_frame)
        if not os.path.exists(filename):
            raise IOError("Save failed")
        print(f"Image saved: {filename}")
        paused = False
        name_frame.pack_forget()
        show_message("✅ Registration successful!", "green")
    except Exception as e:
        print("Error:", e)
        show_message("❌ Registration failed: Could not save image", "red")

def show_message(text, color):
    message_label.config(text=text, fg=color)
    message_label.pack(pady=5)
    root.after(2000, lambda: message_label.pack_forget())

# Camera setup
cap = cv2.VideoCapture(0)
scale = 1.5
paused = False
last_frame = None
model = FaceDetection()

# UI setup
root = tk.Tk()
root.title("Face Registration System")

video_label = tk.Label(root)
video_label.pack()

gamma_slider = tk.Scale(root, from_=10, to=300, orient=tk.HORIZONTAL,
                        label="Gamma x0.01", length=400)
gamma_slider.set(100)
gamma_slider.pack()

# Confirmation frame
confirm_frame = tk.Frame(root)
confirm_label = tk.Label(confirm_frame, text="Do you want to register with this image?", font=("Arial", 12))
confirm_label.pack(pady=5)

yes_btn = tk.Button(confirm_frame, text="Yes", command=on_yes, width=10, bg="green", fg="white")
yes_btn.pack(side=tk.LEFT, padx=10)

no_btn = tk.Button(confirm_frame, text="No", command=on_no, width=10, bg="red", fg="white")
no_btn.pack(side=tk.RIGHT, padx=10)

# Name entry frame
name_frame = tk.Frame(root)
name_label = tk.Label(name_frame, text="Enter name:")
name_label.pack(side=tk.LEFT)

name_entry = tk.Entry(name_frame)
name_entry.pack(side=tk.LEFT, padx=5)

save_btn = tk.Button(name_frame, text="Save", command=save_image, bg="blue", fg="white")
save_btn.pack(side=tk.LEFT, padx=5)

# Message label
message_label = tk.Label(root, font=("Arial", 12, "bold"))

# Key bindings
root.bind("<c>", toggle_pause)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()