import os
import json
import re
from PyPDF2 import PdfReader
import spacy

# Load the small English SpaCy model
nlp = spacy.load("en_core_web_sm")

# Create a directory to store extracted text data if it doesn't exist
os.makedirs('text', exist_ok=True)

# Function to extract text from all pages of a PDF document
def extract_text_from_pdf(pdf_path):
    """Extracts text from all pages of a PDF file."""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        yield page.extract_text()

# Load the database of company data
with open("database.json", 'r') as file:
    company_data = json.load(file)

# Function to check if a sentence is coherent
def is_sentence_coherent(sentence):
    """
    Determines if a sentence is coherent by checking
    for the presence of at least one subject and one verb.
    """
    has_subject = any(token.dep_.endswith("subj") for token in sentence)
    has_verb = any(token.pos_ == "VERB" for token in sentence)
    return has_subject and has_verb

# Iterate through each company and its associated report data
for company_name, reports in company_data.items():
    for report_year in reports:
        print(f"Processing report for {company_name} ({report_year})")

        # Path to the PDF file and its corresponding JSON file 
        pdf_path = os.path.join('data', company_name, f"{report_year}.pdf")
        company_text_dir = os.path.join('text', company_name)
        json_file_path = os.path.join(company_text_dir, f"{report_year}.json")

        # Skip if this report has already been processed
        if os.path.exists(json_file_path):
            print(f"Report {report_year} for {company_name} already processed. Skipping.")
            continue

        # Ensure the directory for storing text data exists
        os.makedirs(company_text_dir, exist_ok=True)

        # Prepare data structure to store extracted and processed text
        processed_data = {'pages': []}

        # Extract and process text from each page of the PDF
        for page_text in extract_text_from_pdf(pdf_path):
            # Replace newlines with spaces for better sentence parsing
            cleaned_text = re.sub(r'\n+', ' ', page_text)
            doc = nlp(cleaned_text)

            # Filter coherent sentences
            coherent_sentences = [
                str(sentence) for sentence in doc.sents if is_sentence_coherent(sentence)
            ]

            # Add coherent sentences to the page data
            if coherent_sentences:
                processed_data['pages'].append(coherent_sentences)

        # Save the processed text data to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(processed_data, json_file)

        print(f"Processed report for {company_name} ({report_year}) saved.")
