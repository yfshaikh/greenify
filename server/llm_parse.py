import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.together_textgen import text_generator as together_text_generator
from llm.monster_textgen import text_generator as monster_text_generator

import os
import json
import tqdm
import threading
import time

# Default to using TogetherAI's text generation
generate_text_function = together_text_generator

# Helper function to generate a structured prompt for analysis
def generate_prompt(page_data):
    return """
You are an analyst extracting key metrics and points from a page in an official document.
Note that you do not care about redundant information or irrelevant points.
You only extract key points that provide useful insights about a company's CSR.

The output should be a list of objects, where each object represents a single point or fact, and contains the following properties:
- `value`: The value of the point or fact, which can be a number, a metric, or a descriptive text.
- `metric`: A boolean indicating whether the value is a metric or not.
- `topic`: The topic of the point or fact, which can be "E" for environmental, "S" for social, or "G" for governance.
- `description`: A brief description of the point or fact, providing context and additional information.
- `tags`: An array of keywords or tags associated with the point or fact, which can be used for filtering or searching.

Input:
```start
{}
end```
Output:
```start""".format(page_data) + "end```"

# Create directory for parsed results if it doesn't exist
if not os.path.exists('parsed'):
    os.mkdir('parsed')

# Load company data
with open("database.json", 'r') as database_file:
    company_data = json.load(database_file)

# Process each company's data
for company_name in company_data.keys():
    for report_year in company_data[company_name]:
        print(f"Processing {company_name}, {report_year}")

        # File paths for input and parsed output
        source_path = os.path.join('text', company_name, f"{report_year}.json")
        destination_path = os.path.join('parsed', company_name, f"{report_year}.json")

        # Skip already processed files
        if os.path.exists(destination_path):
            continue

        # Ensure parent directories exist
        parent_directory = os.path.join('parsed', company_name)
        if not os.path.exists(parent_directory):
            os.mkdir(parent_directory)

        # Initialize parsed data container
        parsed_results = []
        with open(source_path, 'r') as source_file:
            document_data = json.load(source_file)

        # Threaded task to process a page
        def process_page_in_thread(page_content):
            try:
                result = generate_text_function(generate_prompt(page_content))
                parsed_results.append(result)
            except Exception as error:
                print(f"Error processing page: {error}")
                try:
                    time.sleep(1.1)
                    first_half = page_content[:len(page_content) // 2]
                    second_half = page_content[len(page_content) // 2:]
                    result_first = generate_text_function(generate_prompt(first_half))
                    result_second = generate_text_function(generate_prompt(second_half))
                    parsed_results.append([result_first, result_second])
                except Exception as retry_error:
                    print(f"Retry failed for {company_name}, {report_year}: {retry_error}")

        # Manage threading for processing document pages
        threads = []
        num_threads = 4
        for batch_index in tqdm.tqdm(range(len(document_data['pages']) // num_threads)):
            for page in document_data['pages'][batch_index * num_threads:(batch_index + 1) * num_threads]:
                thread = threading.Thread(target=process_page_in_thread, args=(page,))
                threads.append(thread)
                thread.start()
                time.sleep(1.1)
            for thread in threads:
                thread.join()

        # Save parsed results
        with open(destination_path, 'w') as destination_file:
            json.dump({'parsed_pages': parsed_results}, destination_file)
