from huggingface_hub import list_repo_files, hf_hub_download
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- CONFIG ----
repo_id = "lmms-lab/LLaVA-Video-178K"
output_dir = "/data2/local_datasets/LongVQA/hf/hub/LLaVA-Video-178K"
num_workers = 8  # 동시에 다운로드할 쓰레드 수 (네트워크/디스크 속도에 맞게 조정)

# ---- STEP 1: Get all files ----
print("📦 Fetching file list...")
all_files = list_repo_files(repo_id, repo_type="dataset")

# ---- STEP 2: Filter for .tar.gz and .json ----
target_files = [f for f in all_files if f.endswith((".tar.gz", ".json"))]
print(f"✅ Found {len(target_files)} files to download.")

# ---- STEP 3: Define download task ----
def download_file(file_name):
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        return f"✅ {file_name} saved to {file_path}"
    except Exception as e:
        return f"❌ Failed to download {file_name}: {e}"

# ---- STEP 4: Run with ThreadPoolExecutor ----
os.makedirs(output_dir, exist_ok=True)

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(download_file, f): f for f in target_files}
    for future in as_completed(futures):
        result = future.result()
        print(result)

print("🎉 All downloads complete!")
