for dir in /home/qwe/.cache/huggingface/lerobot/ep10/*/; do
    name=$(basename "$dir")
    python convert_dataset_v30_to_v21.py \
        --repo-id="ep10/${name}" \
        --root="$dir"
done