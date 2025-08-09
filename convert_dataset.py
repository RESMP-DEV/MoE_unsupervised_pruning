import csv
import json

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    """
    Converts a two-column (nl, bash) CSV file to a JSONL file
    formatted for the C-Prune script.
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file, \
             open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:

            # Read the CSV file
            reader = csv.DictReader(csv_file)

            # Process each row
            for row in reader:
                nl_command = row.get('nl', '').strip()
                bash_command = row.get('bash', '').strip()

                if not nl_command or not bash_command:
                    continue

                # Combine the columns into a single string
                # This format is similar to how instruction-following models are trained
                combined_text = f"Instruction: {nl_command}\nCommand: {bash_command}"

                # Create a JSON object and write it as a line in the .jsonl file
                json_record = {"text": combined_text}
                jsonl_file.write(json.dumps(json_record) + '\n')

        print(f"Successfully converted {csv_file_path} to {jsonl_file_path}")

    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- USAGE ---
# Replace with the path to your CSV file
your_csv_file = './data/train.csv'
# This will be the output file you use for pruning
output_jsonl_file = './data/pruning_dataset.jsonl'

convert_csv_to_jsonl(your_csv_file, output_jsonl_file)