from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, help='Path to root dataset directory')
    parser.add_argument('--output-csv-path', type=str, default="train.csv", help='Path to output csv')
    args = parser.parse_args()

    root_path = Path(args.data_root)
    pd.DataFrame({"names": [path.name for path in (root_path / "img").glob("*")]}).to_csv(args.output_csv_path,
                                                                                          index=False)