import os
import sys
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import zipfile

def get_input(
    local:bool=False, # Flag to indicate local vs C2D
):
    if local:
        print("Reading local medicaldata directory.")
        # Root directory for dataset
        filename = Path('./data')
        return filename

    dids = os.getenv('DIDS', None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    cwd = os.getcwd()
    print('cwd', cwd)

    did = dids[0]
    print(f"DID: {did}")

    for did in dids:
        filename = Path(f'/data/inputs/{did}/0')

        return filename


def get_df(
    local:bool, # Flag to indicate local vs C2D
):
    print("Preparing df.")
    filename = get_input(local)

    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir()

    data = Path('data')
    if not data.exists():
        data.mkdir()
    try:
        with open(filename) as datafile:
            print('check if its a html')
            print(type(datafile))
            data = datafile.read()
            print(data)
            print('_____________________________________________')
    except:
        print('except - check if its a html')
        print('_____________________________________________')
        pass

    try:
        print('lets see if its unzipped')
        print(type(filename))
        fns = []
        for root, dirs, files in os.walk(str(filename)):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:
                fn = os.path.join(root,file)
                if fn.split('.')[-1] in ['jpeg', 'jpg', 'png']:
                    fns.append(fn)
                print(len(path) * '---', file)

    except:
        pass

    try:
        print('lets unzip and print files')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(str(data))

        print(f"data folder exists: {os.path.exists(str(data))}")

        for root, dirs, files in os.walk(str(data)):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:
                fn = os.path.join(root,file)
                if fn.split('.')[-1] in ['jpeg', 'jpg', 'png']:
                    fns.append(fn)
                print(len(path) * '---', file)
    except:
        pass


def setup_train(
    local:bool, # Flag to indicate local vs C2D
):
    df = get_df(local)


def run(
    local:bool=False
):
    if local:
        print(f"You are in local env")
    if not local:
        print(f"You are in C2D")

    setup_train(local)

if __name__ == "__main__":

    local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run(local)
