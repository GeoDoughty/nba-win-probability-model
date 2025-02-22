import tarfile
import pandas as pd
from io import BytesIO

with tarfile.open(r'data\raw\pbp\zip\pbpstats_2022.tar.xz', mode='r:xz') as tar:
    files = tar.getmembers()
    file_bytes = tar.extractfile(files[0]).read()

pbp_df = pd.read_csv(BytesIO(file_bytes))