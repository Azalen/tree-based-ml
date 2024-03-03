import pathlib
import os
import pandas as pd
import ast
import re

OUTFILE_NAME = "python_analysis.csv"
ROOT_DIR = pathlib.Path("./cfs-sdwan-cisco/python").absolute()
PACKAGE_NAME = "cfs-sdwan-cisco"

def get_python_file_paths(directory: pathlib.Path):
    return list(directory.rglob('*.py'))

def build_out_path() -> pathlib.Path:
    return ROOT_DIR.joinpath(OUTFILE_NAME)

def get_service_name(path: pathlib.Path) -> str:
    return path.stem


def analyse(path: pathlib.Path) -> pd.DataFrame:
    counter_map = {
        ast.Expr.__name__: 0,
        ast.UnaryOp.__name__: 0,
        ast.BinOp.__name__: 0,
        ast.BoolOp.__name__: 0,
        ast.Compare.__name__: 0,
        ast.Call.__name__: 0,
        ast.IfExp.__name__: 0,
        ast.Subscript.__name__: 0,
        ast.ListComp.__name__: 0,
        ast.SetComp.__name__: 0,
        ast.GeneratorExp.__name__: 0,
        ast.DictComp.__name__: 0,
        ast.If.__name__: 0,
        ast.For.__name__: 0,
        ast.While.__name__: 0,
        ast.Try.__name__: 0,
        ast.With.__name__: 0,
        ast.FunctionDef.__name__: 0,
        ast.Lambda.__name__: 0,
        ast.ClassDef.__name__: 0,
        ast.AsyncFunctionDef.__name__: 0,
        ast.Await.__name__: 0,
        ast.AsyncFor.__name__: 0,
        ast.AsyncWith.__name__: 0,
    }

    with open(path, "r+") as file:
        ast_tree = ast.parse(file.read())

    for node in ast.walk(ast_tree):
        node_type = type(node).__name__
        for key in counter_map.keys():
            if key == node_type:
                counter_map[key] += 1

    # rename dict keys to better values
    table = dict()
    # add init service_name column
    table["service_name"] = get_service_name(path)
    for key in counter_map.keys():
        data = counter_map[key]
        label_core = re.sub(r"(?<!^)(?=[A-Z])", "-", key).lower()
        label = f"python-{label_core}-count"
        # create new column in map
        table[label] = [data]

    return pd.DataFrame(table)


def main():
    python_files = get_python_file_paths(ROOT_DIR)
    combined_df = pd.DataFrame()

    for file in python_files:
        df = analyse(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Aggregate the data and add it as a final row
    aggregated_data = combined_df.sum(numeric_only=True)
    aggregated_data['service_name'] = PACKAGE_NAME + "-all"
    aggregated_row = pd.DataFrame([aggregated_data])  # Convert to DataFrame
    combined_df = pd.concat([combined_df, aggregated_row], ignore_index=True)

    # Round all decimal values to 2 digits
    combined_df = combined_df.round(2)

    # Save the data to a CSV file
    combined_df.to_csv(build_out_path(), index=False)

if __name__ == "__main__":
    main()
