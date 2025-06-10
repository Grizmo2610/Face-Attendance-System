import os
import shutil
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition CLI Tool")

    parser.add_argument("--log-level", default='i', choices=['d', 'i', 'w'],
                        help="Log level: d=DEBUG, i=INFO, w=WARNING (default: i)")
    parser.add_argument("--no-console", action="store_true",
                        help="Disable log output to console")
    parser.add_argument("--db-path", default="face_db.pkl",
                        help="Path to database file inside 'database/' folder (default: face_db.pkl)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register
    register_parser = subparsers.add_parser("register", help="Register a new face")
    register_parser.add_argument("--name", required=True, help="Name of the person")
    register_parser.add_argument("--image", required=True, help="Path to image file")
    register_parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold")

    # Verify
    verify_parser = subparsers.add_parser("verify", help="Verify a face")
    verify_parser.add_argument("--image", required=True, help="Path to image file")
    verify_parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")

    # Rename
    rename_parser = subparsers.add_parser("rename", help="Rename a registered user")
    rename_parser.add_argument("--old", required=True, help="Old name")
    rename_parser.add_argument("--new", required=True, help="New name")

    # Reset
    subparsers.add_parser("reset", help="Clear database, logs, and sample images")

    # Import Excel
    import_parser = subparsers.add_parser("import_excel", help="Import faces from Excel file")
    import_parser.add_argument("--file", required=True, help="Path to Excel file (.xls or .xlsx)")

    args = parser.parse_args()

    log_level = parse_log_level(args.log_level)
    show_console = not args.no_console
    db_path = args.db_path

    fd = FaceDetection(database_path=db_path, log_level=log_level, log_to_console=show_console)

    try:
        if args.command == "register":
            fd.register_face(name=args.name, image_path=args.image, threshold=args.threshold)
        elif args.command == "verify":
            fd.verify_face(source=args.image, threshold=args.threshold)
        elif args.command == "rename":
            fd.rename_user(old_name=args.old, new_name=args.new)
        elif args.command == "reset":
            fd.reset_system()
        else:
            parser.print_help()
    except Exception as e:
        print(f"❌ Error: {e}")
