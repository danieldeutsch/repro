echo '{"name": "French Toast", "ingredients": ["1 slice bread", "1 teaspoon cinnamon", "1 egg"]}' > input.jsonl
python generate_recipes.py \
  --input-file input.jsonl \
  --model-name gpt2-xl \
  --output-file output.jsonl
rm input.jsonl output.jsonl