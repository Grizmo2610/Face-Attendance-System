import os
import shutil
import logging

import pandas as pd
from model import FaceDetection

def reset(fd: FaceDetection):
    for folder in [fd.SAMPLE_FOLDER, fd.DATABASE_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    print("✅ Reset completed: all folders removed.")

def parse_log_level(short_level: str) -> int:
    return {
        'd': logging.DEBUG,
        'i': logging.INFO,
        'w': logging.WARNING
    }.get(short_level.lower(), logging.INFO)


def is_valid_excel_file(filename: str) -> bool:
    return filename.lower().endswith(('.xls', '.xlsx'))

def import_from_excel(fd: FaceDetection):
    filepath = input("Enter Excel file path: ").strip()
    if not os.path.exists(filepath):
        print("❌ File does not exist.")
        return
    if not is_valid_excel_file(filepath):
        print("❌ Invalid file format. Only .xls or .xlsx allowed.")
        return

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"❌ Failed to read Excel file: {e}")
        return

    required_cols = {'id', 'name', 'image_path'}
    if not required_cols.issubset(df.columns.str.lower()):
        print(f"❌ Excel file must contain columns: {required_cols}")
        return

    # Chuẩn hóa tên cột về chữ thường để tránh sai
    df.columns = [col.lower() for col in df.columns]

    # Lấy danh sách id hiện tại trong database
    existing_ids = set(fd.get_all_ids())  # Cần hàm này trong FaceDetection, trả về danh sách id hiện có (nếu không có phải bổ sung)
    max_id = max(existing_ids) if existing_ids else 0

    def valid_id(val):
        try:
            v = int(val)
            return v > 0
        except:
            return False

    for index, row in df.iterrows():
        raw_id = row['id']
        name = str(row['name']).strip()
        image_path = str(row['image_path']).strip()

        if not name or not image_path:
            print(f"❌ Row {index+2}: name hoặc image_path bị thiếu, bỏ qua")
            continue

        # Kiểm tra id hợp lệ và chưa tồn tại
        if valid_id(raw_id) and int(raw_id) not in existing_ids:
            assigned_id = int(raw_id)
        else:
            max_id += 1
            assigned_id = max_id
            print(f"⚠️ Row {index+2}: id '{raw_id}' không hợp lệ hoặc đã tồn tại, đổi thành id mới: {assigned_id}")

        # Gọi hàm register (cần sửa FaceDetection để nhận id, hoặc lưu id theo cách riêng)
        try:
            fd.register_face(name=name, image_path=image_path, threshold=0.8, user_id=assigned_id)
            existing_ids.add(assigned_id)
        except Exception as e:
            print(f"❌ Row {index+2}: lỗi đăng ký mặt: {e}")

    print("✅ Import from Excel completed.")
    
def main():
    print("Face Recognition CLI Tool (interactive mode)")
    log_level_input = input("Choose log level (d=DEBUG, i=INFO, w=WARNING) [default i]: ").strip().lower()
    log_level = parse_log_level(log_level_input if log_level_input else 'i')
    no_console_input = input("Disable console logging? (y/n) [default n]: ").strip().lower()
    show_console = (no_console_input != 'y')
    db_path = input("Database file path [default face_db.pkl]: ").strip() or "face_db.pkl"

    fd = FaceDetection(database_path=db_path, log_level=log_level, log_to_console=show_console)

    while True:
        print("\nCommands:")
        print("  1. register     - Register a new face")
        print("  2. verify       - Verify a face")
        print("  3. rename       - Rename a registered user")
        print("  4. reset        - Clear database, logs, and sample images")
        print("  5. import_excel - Import faces from Excel file")
        print("  6. exit         - Exit the program")

        cmd = input("Enter command number: ").strip()

        try:
            if cmd == '1' or cmd.lower() == 'register':
                name = input("Enter name: ").strip()
                image = input("Enter image path: ").strip()
                threshold_input = input("Enter similarity threshold [default 0.8]: ").strip()
                threshold = float(threshold_input) if threshold_input else 0.8
                fd.register_face(name=name, image_path=image, threshold=threshold)
            elif cmd == '2' or cmd.lower() == 'verify':
                image = input("Enter image path: ").strip()
                threshold_input = input("Enter similarity threshold [default 0.6]: ").strip()
                threshold = float(threshold_input) if threshold_input else 0.6
                fd.verify_face(source=image, threshold=threshold)
            elif cmd == '3' or cmd.lower() == 'rename':
                old_name = input("Enter old name: ").strip()
                new_name = input("Enter new name: ").strip()
                fd.rename_user(old_name=old_name, new_name=new_name)
            elif cmd == '4' or cmd.lower() == 'reset':
                reset(fd)
            elif cmd == '5' or cmd.lower() == 'import_excel':
                import_from_excel(fd)
            elif cmd == '6' or cmd.lower() == 'exit':
                print("Exiting program.")
                break
            else:
                print("Invalid command. Please try again.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()