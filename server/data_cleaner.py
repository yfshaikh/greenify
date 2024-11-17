import os
import json
import tqdm

# Create the 'cleaned' directory if it doesn't exist
if not os.path.exists('cleaned'):
    os.mkdir('cleaned')

# Load the dataset from the database file
with open("database.json", 'r') as database_file:
    dataset = json.load(database_file)

def extract_substring(text, start_marker, end_marker):
    # Locate the start marker
    start_pos = text.find(start_marker)
    if start_pos == -1:
        return None  # Start marker not found

    # Adjust position to the actual start of the substring
    start_pos += len(start_marker)

    # Locate the end marker
    end_pos = text.find(end_marker, start_pos)
    if end_pos == -1:
        return None  # End marker not found

    # Extract and return the substring
    return text[start_pos:end_pos]

# Process each company's data
for company_name in dataset.keys():
    for report_year in dataset[company_name]:
        print(f"Processing {company_name}, {report_year}")

        # Define paths for input and cleaned output
        source_path = os.path.join('parsed', company_name, f"{report_year}.json")
        cleaned_output_path = os.path.join('cleaned', company_name, f"{report_year}.json")

        # Skip processing if the cleaned file already exists
        if os.path.exists(cleaned_output_path):
            continue

        # Ensure the parent directory for cleaned data exists
        cleaned_parent_dir = os.path.join('cleaned', company_name)
        if not os.path.exists(cleaned_parent_dir):
            os.mkdir(cleaned_parent_dir)

        # Initialize container for cleaned data
        cleaned_data = []

        # Load parsed data
        with open(source_path, 'r') as parsed_file:
            parsed_pages = json.load(parsed_file)

        # Process pages and extract cleaned data
        entry_id = 0
        for page_content in parsed_pages['parsed_pages']:
            if isinstance(page_content, list):
                combined_entries = []
                for sub_page_content in page_content:
                    extracted = extract_substring(sub_page_content, '```start', 'end```')
                    if extracted:
                        try:
                            parsed = json.loads(extracted)
                            for item in parsed:
                                item["id"] = f"{report_year}.{entry_id}"
                                entry_id += 1
                            combined_entries.extend(parsed)
                        except json.JSONDecodeError:
                            pass
                cleaned_data.extend(combined_entries)
            else:
                extracted = extract_substring(page_content, '```start', 'end```')
                if extracted:
                    try:
                        parsed = json.loads(extracted)
                        for item in parsed:
                            item["id"] = f"{report_year}.{entry_id}"
                            entry_id += 1
                        cleaned_data.extend(parsed)
                    except json.JSONDecodeError:
                        pass

        # Save cleaned data to the output file
        with open(cleaned_output_path, 'w') as cleaned_file:
            json.dump(cleaned_data, cleaned_file)
