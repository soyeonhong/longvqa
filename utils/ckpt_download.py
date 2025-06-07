import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    # Argparser for model repo ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, help="Model repo ID on HuggingFace")
    parser.add_argument("--root_dir", type=str, default="/data/soyeon/longvqa/repos/lmms-eval/checkpoints", help="Root directory to save the model")
    args = parser.parse_args()

    # Extract parent name from repo_id
    repo_id = args.repo_id  # e.g., "openai/clip-vit-large-patch14-336"
    repo_name = repo_id.split("/")[-1]  # "clip-vit-large-patch14-336"

    # Construct local_dir path
    base_dir = Path(args.root_dir)
    local_dir = base_dir / repo_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download the model snapshot
    repo_path = snapshot_download(repo_id=repo_id, local_dir=local_dir)

    print("Repo downloaded to:", repo_path)

if __name__ == "__main__":
    main()
