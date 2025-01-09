import os
import pickle
import zipfile
import pandas as pd


def load_model(model_path: str, model_file: str) -> object:
    os.chdir(model_path)
    with zipfile.ZipFile(model_file + ".zip", "r") as zip_ref:
        with zip_ref.open(model_file + ".pkl") as file:
            loaded_model = pickle.load(file)
    return loaded_model


def load_and_dateindex_data(file_path: str, file_name: str) -> pd.DataFrame:
    os.chdir(file_path)
    df = pd.read_parquet(file_name)

    df.set_index("Date", inplace=True)
    df.sort_index(ascending=True, inplace=True)

    return df


def save_model(model: object, model_path: str, model_file: str) -> None:
    os.chdir(model_path)
    pickle_file = model_file + ".pkl"
    zip_file = model_file + ".zip"

    with open(pickle_file, "wb") as file:
        pickle.dump(model, file)

    # Zip the pickle file
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(pickle_file, compress_type=zipfile.ZIP_DEFLATED)

    # Optionally, remove the pickle file after zipping
    os.remove(pickle_file)
