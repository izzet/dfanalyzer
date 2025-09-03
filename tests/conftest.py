import glob
import json
import os
import pytest
import tarfile


@pytest.fixture(scope='session', autouse=True)
def extract_test_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    tar_files = glob.glob(os.path.join(data_dir, '*.tar.gz'))

    for tar_path in tar_files:
        tar_name = os.path.basename(tar_path)
        extract_folder_name = tar_name.replace('.tar.gz', '')
        extract_path = os.path.join(data_dir, 'extracted', extract_folder_name)

        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        if not any(os.scandir(extract_path)):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)


@pytest.fixture(scope="session")
def epoch_posix_events():
    """Return up to 3 epochs extracted from a real trace file.

    Each epoch is a list of the original JSON lines (strings) starting with an
    epoch.start metadata message, containing some POSIX events in between, and
    finishing with an epoch.end metadata message.
    """
    base = os.path.join(os.path.dirname(__file__), "data", "extracted", "dftracer-ai-logging")
    file_path = os.path.join(base, "trace-0-of-8.pfw")

    if not os.path.exists(file_path):
        # If the test data isn't present, skip tests that request this fixture.
        raise FileNotFoundError(f"Real trace file not found: {file_path}")

    epochs = []
    events = None
    posix_count = 0
    with open(file_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                # ignore non-json lines
                continue

            # detect metadata epoch.start (example: ph == 'M' and args.name == 'epoch.start')
            if j.get("ph") == "M" and j.get("args", {}).get("name") == "epoch.start":
                # start a new epoch capture; use a plain list for events
                events = [line]
                posix_count = 0
                continue

            if events is not None:
                # detect epoch.end metadata; always include it and close the epoch
                if j.get("ph") == "M" and j.get("args", {}).get("name") == "epoch.end":
                    events.append(line)
                    epochs.append(events)
                    events = None
                    posix_count = 0
                    continue

                # Only include POSIX events (cat == 'POSIX'), limit to first 100
                if j.get("cat") == "POSIX":
                    if posix_count < 100:
                        events.append(line)
                        posix_count += 1
                    # else: drop additional POSIX events beyond 100

            # stop when we have three epochs
            if len(epochs) >= 3:
                break

    return epochs
