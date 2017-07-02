import os


def create_dir_if_needed(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def open_creating_dir_if_needed(filename, mode):
    create_dir_if_needed(filename)
    return open(filename, mode)
