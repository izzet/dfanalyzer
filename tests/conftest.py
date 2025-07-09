import glob
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
