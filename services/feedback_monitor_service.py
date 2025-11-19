import os
import sys
import json
import time
import requests
from pinecone import Pinecone
from urllib.parse import urlparse

# --- 1. Setup Project Paths ---
# This allows the service to import modules from other directories
try:
    # Get the directory of this script
    SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Get the project root (one level up)
    PROJECT_ROOT = os.path.dirname(SERVICE_DIR)
except NameError:
    # Fallback for interactive environments
    SERVICE_DIR = os.path.abspath("")
    PROJECT_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, ".."))

# Add project root to the Python path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print(f"Project Root added to path: {PROJECT_ROOT}")

# --- 2. Import Your Existing Indexing Functions ---
try:
    from indexing.index import (
        get_pinecone_credentials_from_json,
        initialize_pinecone_indexes,
        process_and_upload_parent_child
    )
    print("Successfully imported indexing functions.")
except ImportError as e:
    print(f"Error: Could not import from 'indexing.index'. Make sure you are running this from the 'services' directory. {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- 3. Define Service Constants ---
LOG_FILE = os.path.join(PROJECT_ROOT, "feedback_log.jsonl")
STATE_FILE = os.path.join(SERVICE_DIR, "monitor_state.json")
DOWNLOAD_DIR = os.path.join(SERVICE_DIR, "temp_downloads")
ASSETS_FILE = os.path.join(PROJECT_ROOT, "assets.json")
CHECK_INTERVAL_SECONDS = 300  # 5 minutes

# Pinecone Index Names (must match indexing/index.py)
CHILD_INDEX_NAME = "sme-agent-child-chunks"
PARENT_INDEX_NAME = "sme-agent-parent-chunks"

# --- 4. Helper Functions ---

def get_last_processed_timestamp(state_file: str) -> str:
    """Reads the timestamp of the last processed log entry."""
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            return state.get("last_processed_timestamp", "1970-01-01T00:00:00.000000")
    except FileNotFoundError:
        return "1970-01-01T00:00:00.000000"
    except json.JSONDecodeError:
        print(f"Warning: Could not decode state file {state_file}. Starting from scratch.")
        return "1970-01-01T00:00:00.000000"

def set_last_processed_timestamp(state_file: str, timestamp: str):
    """Writes the timestamp of the most recent processed log entry."""
    try:
        with open(state_file, 'w') as f:
            json.dump({"last_processed_timestamp": timestamp}, f)
    except Exception as e:
        print(f"Error: Could not write to state file {state_file}: {e}")

def download_file(url: str, save_dir: str) -> str | None:
    """Downloads a file from a URL to a local directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        print(f"Downloading new source from {url}...")
        
        # Use a single stream request to get headers and content
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            
            # --- START OF FIX ---
            content_type = r.headers.get('content-type', '').lower()
            
            # Get filename from URL path
            parsed_path = urlparse(url).path
            filename = os.path.basename(parsed_path)

            # If path gives no filename (e.g., "page.com/"), use the fallback
            if not filename:
                filename = f"downloaded_{int(time.time())}"

            # If it's a webpage, save as .txt so load_document can read it
            if 'text/html' in content_type and not (filename.endswith('.txt') or filename.endswith('.md')):
                # Remove any existing bad extension (like .com or /)
                filename = os.path.splitext(filename)[0] + ".txt"
            
            # --- END OF FIX ---

            local_filepath = os.path.join(save_dir, filename)
            print(f"Saving to {local_filepath}...")

            # Now download the content
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print("Download complete.")
        return local_filepath
        
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not download file from {url}. {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to save downloaded file. {e}")
        return None

def is_url(path: str) -> bool:
    """Simple check to see if a string is a URL."""
    return path.startswith("http://") or path.startswith("https://")

# --- 5. Main Service Logic ---

def check_feedback_log(pc: Pinecone, child_index, parent_index):
    """
    Reads the feedback log, finds new entries with 'new_source',
    and triggers the indexing pipeline for them.
    """
    print(f"\nChecking for new feedback entries in {LOG_FILE}...")
    if not os.path.exists(LOG_FILE):
        print("Log file not found. Waiting for it to be created...")
        return

    last_ts = get_last_processed_timestamp(STATE_FILE)
    new_entries = []

    # 1. Read the log file
    try:
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry['timestamp'] > last_ts:
                        new_entries.append(entry)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping corrupt log line: {line}")
    except Exception as e:
        print(f"Error: Could not read log file {LOG_FILE}. {e}")
        return

    if not new_entries:
        print("No new feedback entries found.")
        return

    print(f"Found {len(new_entries)} new feedback entries. Processing...")
    
    # Sort entries by timestamp just in case
    new_entries.sort(key=lambda x: x['timestamp'])

    # 2. Filter for new sources and process them
    for entry in new_entries:
        new_source = entry.get("new_source")
        
        if new_source:
            print(f"\nProcessing new source from feedback (Timestamp: {entry['timestamp']}): {new_source}")
            
            local_filepath = None
            
            if is_url(new_source):
                # 3a. It's a URL, download it
                local_filepath = download_file(new_source, DOWNLOAD_DIR)
            elif os.path.exists(new_source):
                # 3b. It's a local file path that exists
                print("Source is a valid local file.")
                local_filepath = new_source
            else:
                print(f"Warning: Source '{new_source}' is not a valid URL or existing local file. Skipping.")
            
            if local_filepath:
                try:
                    # 4. Trigger the existing indexing pipeline
                    print(f"Starting indexing for: {local_filepath}")
                    process_and_upload_parent_child(
                        file_path=local_filepath,
                        child_index=child_index,
                        parent_index=parent_index,
                        parent_dir=PROJECT_ROOT  # Pass the project root
                    )
                    print(f"Successfully indexed new source: {local_filepath}")
                except Exception as e:
                    print(f"Error: Failed to process and index '{local_filepath}'. {e}")

        # 5. Update the state file with the timestamp of the last processed entry
        set_last_processed_timestamp(STATE_FILE, entry['timestamp'])
        
    print("Feedback processing complete.")


# --- 6. Main Service Loop ---
def main():
    print("--- Starting Feedback Monitor Service ---")
    
    # 1. Initialize Pinecone
    try:
        PINECONE_API_KEY = get_pinecone_credentials_from_json(ASSETS_FILE)
        if not PINECONE_API_KEY:
            raise ValueError("Could not find 'pinecone_api_key' in the assets.json file.")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        child_index, parent_index = initialize_pinecone_indexes(
            pc, CHILD_INDEX_NAME, PARENT_INDEX_NAME
        )
        print("Pinecone indexes initialized successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not initialize Pinecone. {e}")
        sys.exit(1)

    # 2. Run the service loop
    try:
        while True:
            check_feedback_log(pc, child_index, parent_index)
            print(f"Sleeping for {CHECK_INTERVAL_SECONDS} seconds...")
            time.sleep(CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n--- Shutting down Feedback Monitor Service ---")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        print("Service stopped.")

if __name__ == "__main__":
    main()