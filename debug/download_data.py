import os
import json
from datasets import load_dataset

# check if ./tinystories_raw.jsonl exist already, if so dont download

# if not os.path.exists("./download/tinystories_raw.jsonl"):
#     print("Downloading TinyStories dataset...")
#     # Load the dataset
#     dataset = load_dataset("roneneldan/TinyStories")

#     # Open the output file
#     with open("./download/tinystories_raw.jsonl", "w") as f:
#         # Iterate over each split in the dataset
#         for split in dataset.keys():
#             # Iterate over each example in the split
#             for example in dataset[split]:
#                 # Create a document dictionary
#                 document = {"text": example["text"]}
#                 # Write the document to the file
#                 f.write(json.dumps(document) + "\n")
# else:
#     print("TinyStories dataset already downloaded.")

from transformers import LlamaForCausalLM, AutoTokenizer
print("Downloading Llama3 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=True)
tokenizer.save_pretrained("./download/llama3_tokenizer")
print("Downloading Llama3 model...")
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=True)
model.save_pretrained("./download/llama3_model")