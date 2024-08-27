python ../tools/preprocess_data.py --input ./download/tinystories_raw.jsonl \
	--tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ./download/llama3_tokenizer \
	--output-prefix ./download/tokenized_dataset \
	--workers=16