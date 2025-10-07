import csv

# Input and output file paths
input_path = 'augmented_data.csv'
output_path = 'filtered_human_responses.csv'

# Step 1: Find all convIDs with label 'Gemma3_4b' or 'Dolphin-Mistral'
relevant_convids = set()
with open(input_path, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    for row in reader:
        if len(row) >= 3 and row[2].strip() in {'Gemma3_4b', 'Dolphin-Mistral'}:
            relevant_convids.add(row[0].strip())

# Step 2: Write only 'Human' responses with those convIDs
with open(input_path, newline='', encoding='utf-8') as infile, \
     open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        if len(row) >= 3 and row[2].strip() == 'Human' and row[0].strip() in relevant_convids:
            writer.writerow(row)

print(f"Filtered responses written to {output_path}")
