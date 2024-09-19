import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Run distributed training for GPT model")
    parser.add_argument("--tp", type=int, default=1, help="Tensor model parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline model parallel size")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of the node")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master_port", type=str, default="8000", help="Master node port")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Directory containing the tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the tokenized dataset")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory containing the model weights")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    # Construct the distributed arguments
    distributed_args = [
        "--nproc_per_node", str(args.nproc_per_node),
        "--nnodes", str(args.nnodes),
        "--node_rank", str(args.node_rank),
        "--master_addr", args.master_addr,
        "--master_port", args.master_port
    ]
    
    # Construct the command
    if args.debug:
        cmd = ["debugpy-run", "-m", "torch.distributed.run", "--"] +  distributed_args + ["../pretrain_gpt.py"]
    else:
        cmd = ["torchrun"] + distributed_args + ["../pretrain_gpt.py"]

    cmd += [
        "--tensor-model-parallel-size", str(args.tp),
        "--pipeline-model-parallel-size", str(args.pp),
        "--seq-length", "1024",
        "--max-position-embeddings", "1024",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", args.tokenizer_model,
        "--load", args.load_dir,
        "--exit-on-missing-checkpoint",
        "--use-checkpoint-args",
        "--no-load-optim",
        "--no-load-rng",
        "--untie-embeddings-and-output-weights",
        "--normalization", "RMSNorm",
        "--position-embedding-type", "rope",
        "--no-masked-softmax-fusion",
        "--attention-softmax-in-fp32",
        "--disable-bias-linear",
        "--transformer-impl", "transformer_engine",
        "--group-query-attention",
        "--num-query-groups", "8",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--rotary-base", "500000",
        "--rotary-percent", "1.0",
        "--ffn-hidden-size", "14336",
        "--num-attention-heads", "32",
        "--swiglu",
        "--bf16",
        "--no-gradient-accumulation-fusion",
        "--micro-batch-size", "1",
        "--no-async-tensor-model-parallel-allreduce",
        "--lr", "0.0003",
        "--train-iters", "10",
        "--eval-iters", "0",
        "--log-interval", "1",
        "--data-path", args.data_path,
    ]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()