import tarfile
import pandas as pd
from io import BytesIO
from pathlib import Path

def extract_pbp_from_tar(
    tar_path: Path,
    output_path: Path
) -> pd.DataFrame:
    with tarfile.open(tar_path, mode='r:xz') as tar:
        files = tar.getmembers()
        file_bytes = tar.extractfile(files[0]).read()

    pbp_df = pd.read_csv(BytesIO(file_bytes))
    pbp_df.to_parquet(output_path)


def extract_all_pbp_files(
    tar_folder: Path,
    output_folder: Path
) -> None:
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    for tar_path in tar_folder.glob("*.tar.xz"):
        output_path = output_folder / f"{tar_path.stem}.parquet"
        if output_path.exists():
            continue
        extract_pbp_from_tar(tar_path, output_path)

if __name__ == '__main__':
    extract_all_pbp_files(
        Path("data/raw/pbp/zip"),
        Path("data/raw/pbp/parquet")
    )