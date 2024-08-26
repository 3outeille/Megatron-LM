import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Convert LLaMA/Mistral model to Megatron format")
    parser.add_argument("--tp", type=int, default=1, help="Target tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Target pipeline parallel size")
    parser.add_argument("--load-dir", type=str, required=True, help="Directory containing the HuggingFace model")
    parser.add_argument("--tokenizer-model", type=str, required=True, help="Directory containing the tokenizer")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    # Create the save directory name based on tp and pp values
    save_dir = f"save_megatron_weights_TP{args.tp}_PP{args.pp}"
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    if args.debug:
        cmd = ["debugpy-run", "-m", "torch.distributed.run", "../tools/checkpoint/convert.py", "--"] 
    else:
        cmd = ["python",  "../tools/checkpoint/convert.py"]
        
    # Construct the command
    cmd  += [
        "--bf16",
        "--model-type", "GPT",
        "--loader", "llama_mistral",
        "--saver", "mcore",
        "--target-tensor-parallel-size", str(args.tp),
        "--target-pipeline-parallel-size", str(args.pp),
        "--checkpoint-type", "hf",
        "--load-dir", args.load_dir,
        "--save-dir", save_dir,
        "--tokenizer-model", args.tokenizer_model,
        "--model-size", "llama3-8B"
    ]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Conversion completed successfully. Weights saved in {save_dir}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    main()