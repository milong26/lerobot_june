for dir in /home/qwe/.cache/huggingface/lerobot/robotwin/click_alarmclock/*/; do
    name=$(basename "$dir")
    python convert_dataset_v30_to_v21.py \
        --repo-id="click_alarmclock/${name}" \
        --root="$dir"
done