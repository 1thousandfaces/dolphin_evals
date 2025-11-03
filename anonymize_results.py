import json
import os

def anonymize_jsonl(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                if 'benchmark' in data and isinstance(data['benchmark'], dict) and 'task_results' in data['benchmark']:
                    data['benchmark']['task_results'] = []
                outfile.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                pass

def main():
    input_dir = 'results'
    output_dir = 'results_anonymized'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            anonymize_jsonl(input_path, output_path)
            print(f"Anonymized {input_path} to {output_path}")

if __name__ == '__main__':
    main()
