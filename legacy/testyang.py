import pathlib
import os
import pandas as pd
from yang_api.compiler.parser import parser as yang_parser

OUTFILE_NAME = "yang_analysis.csv"
ROOT_DIR = pathlib.Path("./").absolute()
PACKAGE_NAME = "cfs-sdwan-cisco"

def build_yang_paths():
    yang_dir = ROOT_DIR.joinpath(PACKAGE_NAME + "/src/yang")
    print(yang_dir)
    return list(yang_dir.glob("*.yang"))

def build_out_path() -> pathlib.Path:
    return ROOT_DIR.joinpath(OUTFILE_NAME)

def get_service_name(path: pathlib.Path) -> str:
    return path.stem

def analyse(path: pathlib.Path) -> pd.DataFrame:
    counter_map = {
        "leaf": 0,
        "list": 0,
        "must": 0,
        "mandatory": 0,
        "pattern": 0,
        "container": 0,
        "leaf-list": 0,
        "all-statements": 0,
    }
    special_counter_map = {"avg-pattern-length": 0, "avg-must-length": 0}

    must_statement_lengths = []
    pattern_statement_lengths = []

    yang_tree = yang_parser.tree_from_file(path)

    for node in yang_tree.walk():
        if node.keyword in counter_map.keys():
            counter_map[node.keyword] += 1
        if node.keyword == "must":
            must_statement_lengths.append(len(node.value))
        elif node.keyword == "pattern":
            pattern_statement_lengths.append(len(node.value))
        counter_map["all-statements"] += 1

    if must_statement_lengths:
        special_counter_map["avg-must-length"] = sum(must_statement_lengths) / len(
            must_statement_lengths
        )
    if pattern_statement_lengths:
        special_counter_map["avg-pattern-length"] = sum(
            pattern_statement_lengths
        ) / len(pattern_statement_lengths)

    table = dict()
    table["service_name"] = get_service_name(path)
    for key in counter_map.keys():
        data = counter_map[key]
        label_core = key
        label = f"yang-{label_core}-count"
        # create new column in map
        table[label] = [data]

    for key in special_counter_map.keys():
        data = special_counter_map[key]
        label_core = key
        label = f"yang-{label_core}"
        # create new column in map
        table[label] = [data]

    return pd.DataFrame(table)


def main():
    yang_files = build_yang_paths()
    combined_df = pd.DataFrame()

    for file in yang_files:
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