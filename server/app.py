from flask import Flask, request, jsonify
from rag import RAG
from dotenv import load_dotenv
import os
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_together import Together
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import OpenAI
import json

# Load environment variables
load_dotenv()


# initialize LLM and embeddings
llm = OpenAI(temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings()

# Create a Pinecone instance 
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"), 
    environment=os.getenv("PINECONE_ENV")
)

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["sustainability_db"]
collection = db["company_metrics"]

# Define the index name
index_name = "sustainability-index"

# Create or connect to the Pinecone index
try:
    # Check if the index exists
    if index_name in pc.list_indexes().names():
        index = pc.index(index_name)
    else:
        # Create a new index if it doesn't exist
        index = pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
except Exception as e:
    print(f"Error: {e}")


rag_instance = RAG("CBRE")

def get_metrics_from_mongo(topic, phrase):
    # Query MongoDB for sustainability metrics related to topic and phrase
    result = collection.find_one({"tags": {"$regex": phrase, "$options": "i"}, "metric": True, "topic": topic})
    
    if result:
        return {
            "value": result["value"],
            "description": result["description"]
        }
    return {}

def get_closest_feature_from_pinecone(query, topic):
    # Convert the query to an embedding
    query_embedding = embeddings.embed_query(query)
    
    # Query Pinecone for the closest match
    response = index.query(query_embedding, top_k=1, filter={"metric": True, "topic": topic})
    
    if response and response["matches"]:
        match = response["matches"][0]
        # Fetch metadata from Pinecone match
        return {
            "value": match["metadata"]["value"],
            "description": match["metadata"]["description"]
        }
    
    return {}

def generate_follow_up_questions(context):
    prompt = f"""
    You are an agent speaking with a representative from CBRE.
    Your goal is to assist the representative to help them better understand tenant building emissions and sustainability practices.
    
    You provide follow-up questions that the representative may ask.
    Use the data given to you to provide two follow-up questions they could ask to better understand their data.
    
    You provide your output in JSON format, for example:
    ```
    [
        "What steps can we take to reduce tenant building emissions by 10% over the next year?",
        "How can we improve our energy efficiency in tenant buildings to meet sustainability goals?"
    ]
    ```

    Given the following data, provide questions that the representative may ask:
    ```
    {context}
    ```
    """
    
    result = None
    while not result:
        try:
            output = model(prompt)
            result = json.loads(output[output.index('['):output.index(']')+1])[:2]
        except:
            pass
    return result

def get_cbre_info_in_format():
    return {
        'Environmental': {
            'Tenant Building Emissions': get_metrics_from_mongo('E', 'emissions'),
            'Energy Efficiency': get_metrics_from_mongo('E', 'efficiency'),
            'Renewable Energy': get_metrics_from_mongo('E', 'renewable'),
            'Water Management': get_metrics_from_mongo('E', 'water'),
            'Waste Management': get_metrics_from_mongo('E', 'waste'),
        }
    }

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat_with_bot():
    data = request.json
    user_prompt = data.get('prompt')
    print('===========================')
    print(user_prompt)
    print('===========================')

    # Ensure a prompt is provided
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Convert the query to an embedding
    query_embedding = embeddings.embed_query(user_prompt)

    # Get response from RAG instance using the query_embedding
    response = rag_instance.get_response(query_embedding, user_prompt)

    print('===========================')
    print(response)
    print('===========================')

    # Return the chatbot's response
    return jsonify({"response": response})


@app.route('/reset', methods=['POST'])
@cross_origin()
def reset_conversation():
    global rag_instance
    rag_instance.conversation_history = ""
    return jsonify({"message": "Conversation history reset"})

@app.route('/company_info', methods=['POST'])
@cross_origin()
def company_info():
    data = {}
    data['metadata'] = {
        'companyName': 'CBRE',
        'stockExchange': 'NYSE'
    }
    data['data'] = get_cbre_info_in_format()

    return jsonify(data)

@app.route('/follow_up_questions', methods=['POST'])
@cross_origin()
def follow_up_questions():
    data = request.json
    
    return jsonify({
        'questions': generate_follow_up_questions(get_cbre_info_in_format())
    })

@app.route('/comparison_company_info', methods=['POST'])
@cross_origin()
def comparison_company_info():
    data = request.json
    company_data = data.get('company_data')

    data = {
        'metadata': {
            'companyName': 'CBRE',
            'stockExchange': 'NYSE'
        },
        'data': {}
    }

    if not company_data:
        return jsonify(data)

    for category in company_data['data']:
        data['data'][category] = {}
        for subcategory in company_data['data'][category]:
            data['data'][category][subcategory] = []
            values = company_data['data'][category][subcategory]
            for item in values:
                cf = get_closest_feature_from_pinecone(item['description'], category[0].upper())
                data['data'][category][subcategory].append(cf)

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=False, port=8000)
