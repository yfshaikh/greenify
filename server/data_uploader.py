import os
from dotenv import load_dotenv
import json
import time
import tqdm
import pinecone

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient

# Load environment variables from the .env file
load_dotenv()

# Initialize embeddings using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# MongoDB connection setup
mongo_client = MongoClient()

# Pinecone API initialization
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX_NAME = "company_key_data"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Ensure Pinecone index exists
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=PINECONE_INDEX_NAME, dimension=768)

pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

# MongoDB collection setup
mongo_client.vectordb.drop_collection("company_key_data")

# Load company data from the database
with open("database.json", 'r') as database_file:
    company_data = json.load(database_file)

# Batch size for uploads
UPLOAD_BATCH_SIZE = 1

# Process each company's data
for company_name in company_data.keys():
    for report_year in company_data[company_name]:
        print(f"Processing: {company_name}, {report_year}")
        cleaned_data_path = os.path.join('cleaned', company_name, f"{report_year}.json")

        # Load cleaned data
        with open(cleaned_data_path, 'r') as cleaned_file:
            cleaned_entries = json.load(cleaned_file)

        # Initialize containers for documents and entries
        document_batch = []
        mongo_entries = []

        # Process entries in the cleaned data
        for entry in tqdm.tqdm(cleaned_entries):
            entry['year'] = report_year
            entry['company'] = company_name

            mongo_entries.append(entry)
            vector_id = f"{company_name}-{report_year}-{entry.get('id')}"
            document_batch.append({
                "id": vector_id,
                "values": embedding_model.embed_text(entry["description"]),
                "metadata": entry
            })

            # Upload in batches to Pinecone and MongoDB
            if len(document_batch) >= UPLOAD_BATCH_SIZE:
                # Upload batch to Pinecone
                pinecone_index.upsert(vectors=document_batch)
                document_batch = []

                # Upload batch to MongoDB
                if mongo_entries:
                    mongo_client.vectordb.company_key_data.insert_many(mongo_entries)
                mongo_entries = []

        # Upload remaining entries to Pinecone
        if document_batch:
            pinecone_index.upsert(vectors=document_batch)

        # Upload remaining entries to MongoDB
        if mongo_entries:
            mongo_client.vectordb.company_key_data.insert_many(mongo_entries)

        time.sleep(0.1)
