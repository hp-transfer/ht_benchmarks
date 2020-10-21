"""
wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
tar xf fcnet_tabular_benchmarks.tar.gz
"""
import argparse
import tarfile
import urllib.request

from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="data/fcnet", type=Path)
args = parser.parse_args()

DATA_URL = "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
args.output_dir.mkdir(parents=True)
tar_file = args.output_dir / "fcnet.tar.gz"

print("Starting the download. This might take a few minutes.")

urllib.request.urlretrieve(DATA_URL, tar_file)

with tarfile.open(tar_file) as tar_stream:
    tar_stream.extractall(path=args.output_dir)
