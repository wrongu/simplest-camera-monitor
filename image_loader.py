from pathlib import Path
import time
import re


timestamp_regex = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
)


def is_datetime_named(file: Path) -> bool:
    """
    Check if the file has been renamed based on its name.
    A file is considered renamed if it starts with a timestamp in the format YYYYMMDD_HHMMSS.
    """
    return timestamp_regex.match(file.name) is not None


def get_file_time(file: Path) -> float:
    """Get the file timestamp; in a first pass, this is the file modification time. Files are then
    renamed based on this timestamp."""
    match = timestamp_regex.match(file.name)
    if match:
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        second = int(match.group("second"))
        return time.mktime((year, month, day, hour, minute, second, 0, 0, -1))
    else:
        # If the file is not renamed yet, its timestamp is its last modification time, floored
        # to the nearest second
        return int(file.stat().st_mtime)


def get_all_timestamped_files_sorted(directory: Path, glob="*") -> list[tuple[float, Path]]:
    return sorted([(get_file_time(f), f) for f in directory.glob(glob)])


def rename_all(directory: Path, dry_run: bool):
    for file in directory.glob("**/*.png"):
        if not file.is_file():
            continue

        # Check if the file has a name suggesting it has already been renamed once before
        if is_datetime_named(file):
            continue

        # Get the last modified time of the file
        timestamp = file.stat().st_mtime
        # Convert to a human-readable format
        formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))

        # Create a new name for the file
        new_name = f"{formatted_time}_{file.name}"
        new_path = file.parent / new_name

        if dry_run:
            print(f"mv {file} {new_path}")
        else:
            file.rename(new_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rename files based on their timestamps.")
    parser.add_argument("directory", type=Path, help="Directory containing the files to rename.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform a dry run without renaming files."
    )
    args = parser.parse_args()

    rename_all(args.directory, dry_run=args.dry_run)
