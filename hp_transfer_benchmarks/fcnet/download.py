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
    
    import os
    
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar_stream, path=args.output_dir)
