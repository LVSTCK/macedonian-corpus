""" Download the raw Macedonian corpus from the Hugging Face Hub. """ 

from huggingface_hub import hf_hub_download, login

login(token="<YOUR_TOKEN>")
destination_directory = "."
file_path = hf_hub_download(
    repo_id="LVSTCK/macedonian-corpus-raw",
    filename="corpus-mk.jsonl.gz",  
    repo_type="dataset",  
    local_dir=destination_directory,  
)

print(f"File downloaded to: {file_path}")
