import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


# 1. Temporary directory that auto-cleans up
@contextmanager
def temp_directory():
    """Create a temporary directory and clean it up when done."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Usage:
# with temp_directory() as tmp_dir:
#     (tmp_dir / "test.txt").write_text("Hello world")
#     # Directory automatically deleted when exiting


# 2. Timer context manager
@contextmanager
def timer(label="Operation"):
    """Time how long a block of code takes to execute."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{label} took {end_time - start_time:.2f} seconds")


# Usage:
# with timer("Database query"):
#     # Some slow operation
#     time.sleep(1)


# 3. Suppress stdout/stderr
@contextmanager
def suppress_output(encoding: str):
    """Suppress all stdout and stderr output."""
    with open(os.devnull, "w", encoding=encoding) as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# Usage:
# with suppress_output():
#     print("This won't be printed")


# 4. Environment variable context manager
@contextmanager
def env_var(key, value):
    """Temporarily set an environment variable."""
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


# Usage:
# with env_var("DEBUG", "true"):
#     # Code that uses DEBUG environment variable
#     print(os.environ.get("DEBUG"))  # prints "true"


# 5. File backup context manager
@contextmanager
def backup_file(filepath):
    """Create a backup of a file and restore it if something goes wrong."""
    filepath = Path(filepath)
    backup_path = filepath.with_suffix(filepath.suffix + ".backup")

    # Create backup
    if filepath.exists():
        shutil.copy2(filepath, backup_path)
        backup_created = True
    else:
        backup_created = False

    try:
        yield filepath
    except Exception:
        # Restore backup on exception
        if backup_created:
            shutil.copy2(backup_path, filepath)
        raise
    finally:
        # Clean up backup file
        if backup_created and backup_path.exists():
            backup_path.unlink()


# Usage:
# with backup_file("important.txt") as file:
#     # Modify the file - if an exception occurs, original is restored
#     file.write_text("new content")


# 6. Database transaction context manager
@contextmanager
def database_transaction(connection):
    """Handle database transaction with automatic rollback on error."""
    transaction = connection.begin()
    try:
        yield connection
        transaction.commit()
    except Exception:
        transaction.rollback()
        raise


# Usage:
# with database_transaction(db_conn) as conn:
#     conn.execute("INSERT INTO users ...")
#     conn.execute("UPDATE profiles ...")
#     # Automatically commits on success, rolls back on error


# 7. Capturing stdout
@contextmanager
def capture_stdout():
    """Capture stdout and return it as a string."""
    from io import StringIO

    old_stdout = sys.stdout
    captured_output = StringIO()
    try:
        sys.stdout = captured_output
        yield captured_output
    finally:
        sys.stdout = old_stdout


# Usage:
# with capture_stdout() as output:
#     print("Hello world")
#     print("Another line")
# captured_text = output.getvalue()


# 8. Retry context manager
@contextmanager
def retry_on_failure(max_attempts=3, delay=1):
    """Retry a block of code on failure."""
    for attempt in range(max_attempts):
        try:
            yield attempt + 1
            break  # Success, exit the loop
        except Exception as e:
            if attempt == max_attempts - 1:  # Last attempt
                raise
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


# Usage:
# with retry_on_failure(max_attempts=3, delay=2) as attempt:
#     print(f"Attempt {attempt}")
#     # Some operation that might fail
#     if attempt < 3:
#         raise ValueError("Simulated failure")


# 9. Log level context manager
@contextmanager
def log_level(logger, level):
    """Temporarily change a logger's level."""
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)


# Usage:
# import logging
# logger = logging.getLogger(__name__)
# with log_level(logger, logging.DEBUG):
#     logger.debug("This debug message will be shown")


# 10. Context manager that yields a value
@contextmanager
def managed_resource(resource_name):
    """Simulate acquiring and releasing a resource."""
    print(f"Acquiring {resource_name}")
    resource = f"Resource: {resource_name}"
    try:
        yield resource
    finally:
        print(f"Releasing {resource_name}")


# Usage:
# with managed_resource("database_connection") as resource:
#     print(f"Using {resource}")
#     # Do work with the resource


# 11. Multiple context managers in one
@contextmanager
def combined_context():
    """Combine multiple context managers."""
    with temp_directory() as tmp_dir:
        with timer("Combined operation"):
            yield tmp_dir


# Usage:
# with combined_context() as tmp_dir:
#     # You get both temporary directory and timing


# 12. Working directory with error handling
@contextmanager
def safe_chdir(path):
    """Change directory safely, handling cases where path doesn't exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist")

    oldcwd = os.getcwd()
    os.chdir(path)
    try:
        yield Path(path)
    except Exception:
        # Log the error or handle it as needed
        print(f"Error occurred while in directory {path}")
        raise
    finally:
        os.chdir(oldcwd)


# Usage:
# try:
#     with safe_chdir("/some/path"):
#         # Do work in the directory
#         pass
# except FileNotFoundError as e:
#     print(f"Could not change directory: {e}")
