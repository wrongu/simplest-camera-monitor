from pathlib import Path
import time
import re
import os


timestamp_file_regex = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
)

timestamp_path_regex = re.compile(
    (re.escape(os.path.sep)).join(
        [
            r"(?P<year>\d{4})",
            r"(?P<month>\d{2})",
            r"(?P<day>\d{2})",
            r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})",
        ]
    )
)


def create_timestamped_filename(timestamp: float, suffix: str) -> str:
    """Create a filename based on a timestamp."""
    tmp = time.strftime("%Y____%m____%d____%H%M%S", time.localtime(timestamp)) + suffix
    return tmp.replace("____", os.path.sep)


def is_datetime_named(file: Path) -> bool:
    """
    Check if the file has been renamed based on its name.
    [OLD] A file is considered renamed if it starts with a timestamp in the format YYYYMMDD_HHMMSS.
    [NEW] A file is considered renamed if its name or any part of its path contains a timestamp in the format YYYY/MM/DD/HHMMSS.
    """
    return (
        timestamp_file_regex.match(file.name) is not None
        or timestamp_path_regex.search(str(file)) is not None
    )


def get_file_time(file: Path) -> float:
    """Get the file timestamp; in a first pass, this is the file modification time. Files are then
    renamed based on this timestamp."""
    match = timestamp_path_regex.match(str(file)) or timestamp_file_regex.match(
        file.name
    )
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


def get_all_timestamped_files_sorted(
    directory: Path, glob="**/*.jpg"
) -> list[tuple[float, Path]]:
    return sorted([(get_file_time(f), f) for f in directory.glob(glob)])


def ensure_files_timestamp_named(directory: Path, dry_run: bool, glob="**/*.jpg"):
    for file in directory.glob(glob):
        if not file.is_file():
            continue

        # Check if we need to do any renaming
        rename_timestamp = None
        if not is_datetime_named(file):
            # If file name/path is not in datetime format, use the file modification time
            rename_timestamp = file.stat().st_mtime
        elif timestamp_file_regex.match(file.name):
            # If file is in old-style datetime format, extract the timestamp from the filename
            # and prepare to rename it to the new format
            rename_timestamp = get_file_time(file)

        if rename_timestamp is not None:
            # Convert to a timestamp-path format to create a filename and path from the timestamp
            new_name = create_timestamped_filename(rename_timestamp, file.suffix)
            new_path = file.parent / new_name

            if dry_run:
                if not new_path.parent.exists():
                    print(f"mkdir -p {new_path.parent}")
                print(f"mv {file} {new_path}")
            else:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                file.rename(new_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rename files based on their timestamps."
    )
    parser.add_argument(
        "directory", type=Path, help="Directory containing the files to rename."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without renaming files.",
    )
    args = parser.parse_args()

    ensure_files_timestamp_named(args.directory, dry_run=args.dry_run)
