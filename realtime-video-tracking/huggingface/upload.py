"""Upload the fine-tuned YOLO to the Hugging Face Hub (private by default).

    python huggingface/upload.py                 # create/update the repo, private
    python huggingface/upload.py --public        # make it public

Needs `pip install huggingface_hub` and a login (`hf auth login`) - nothing is stored in this repo.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "weights" / "yolov8s_traffic_best.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=None, help="user/name; default: <your user>/traffic-monitor-yolov8s-visdrone")
ap.add_argument("--public", action="store_true")
a = ap.parse_args()

api = HfApi()
user = api.whoami()["name"]
repo = a.repo or f"{user}/traffic-monitor-yolov8s-visdrone"
assert WEIGHTS.exists(), f"missing {WEIGHTS}"
create_repo(repo, repo_type="model", private=not a.public, exist_ok=True)
api.upload_file(path_or_fileobj=str(WEIGHTS), path_in_repo="yolov8s_traffic_best.pt", repo_id=repo)
api.upload_file(path_or_fileobj=str(HERE / "README.md"), path_in_repo="README.md", repo_id=repo)
api.upload_folder(folder_path=str(HERE / "assets"), path_in_repo="assets", repo_id=repo)
if a.public:
    api.update_repo_settings(repo, private=False)
print("done:", f"https://huggingface.co/{repo}", "(public)" if a.public else "(private)")
