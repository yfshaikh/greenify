from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_together import Together
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from sklearn.decomposition import PCA
import numpy as np
from transformers import pipeline





class RAG:
    def __init__(self, company, model):
        load_dotenv()
        self.llm = OpenAI(temperature=0.9, max_tokens=500)
        # self.embeddings = OpenAIEmbeddings()
        self.embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5")
        self.pca = PCA(n_components=768)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
        
        index_name = "company-key-data"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = pc.Index(index_name)
        self.retriever = self.index.query
        self.model = model
        self.company = company

        # Initialize the summarizer using Hugging Face's pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        self.conversation_history = ""
    
    def update_history(self, question, answer):
        self.conversation_history += f"Representative: {question}\nAgent: {answer}\n"
        history_parts = self.conversation_history.split("\n")
        if len(history_parts) > 6:  # Limiting history to 3 exchanges (6 lines)
            self.conversation_history = "\n".join(history_parts[-6:])
    
    def get_response(self, input_query, user_prompt):
        # Perform a Pinecone query to retrieve the closest matches
        results = self.retriever(
            vector=input_query,
            top_k=5,
            include_values=True,
            include_metadata=True
        )
        
        # Summarize or filter context
        context = self.retrieve_context(user_prompt)
        print("Context retrieved:", context)  # Debugging context
        context = self.summarize_context(context)
        print("Context after summarization:", context)  # Debugging context after summarization

        # Build the entire prompt as a single string
        prompt = f"""
        <s>[INST] You are an agent speaking with a representative from {self.company}.  
        Your goal is to assist the representative with questions regarding {self.company}'s sustainability practices.  
        You are provided with context on {self.company}’s sustainability goals. Focus only on {self.company}’s practices and avoid discussing other companies. Be critical by identifying any missing metrics or areas for improvement, and provide specific, actionable suggestions.  
        Answer the question with a concise, focused response. Split your response into clear, logical paragraphs to improve readability. Use only the provided context to inform your answer. Don't give more than 3 sentences before a line break [/INST]

        Context: {context}

        Conversation History:
        {self.conversation_history}

        Representative: {user_prompt} [/INST]
        """
        
        output = self.llm(prompt)
        
        # Update history
        self.update_history(user_prompt, output)
        
        return output
    
    
    def retrieve_context(self, query):
        # Convert the query into an embedding vector
        query_vector = self.embeddings.embed_query(query)
        
        # Convert the query vector to a NumPy array (if it's not already)
        query_vector = np.array(query_vector)
        
        # Flatten the query vector
        query_vector = query_vector.flatten()

        # Convert the NumPy array to a list for Pinecone serialization
        query_vector_list = query_vector.tolist()

        # Perform a Pinecone query to retrieve relevant documents
        results = self.retriever(
            vector=query_vector_list,  # The embedding vector of the query (as a list)
            top_k=5,  # Number of similar results to fetch
            include_values=True,  # Optionally include the vector values
            include_metadata=True  # Optionally include metadata associated with the results
        )

        # Debugging: print the entire result to inspect the metadata
        print("Pinecone results:", results)

        # Extract the top relevant results (assuming you have metadata or text to return)
        context = ""
        for result in results['matches']:
            # Debugging: print the metadata structure
            print(f"Metadata for result: {result['metadata']}")
            
            # Concatenate relevant fields from metadata to build context
            description = result['metadata'].get('description', 'No description available')
            company = result['metadata'].get('company', 'No company available')
            tags = ", ".join(result['metadata'].get('tags', ['No tags available']))
            
            # Combine the extracted fields into a coherent context
            text = f"Company: {company}\nDescription: {description}\nTags: {tags}\n"
            print(f"Retrieved context: {text}")  # Debugging the context
            context += text + "\n"
        
        print("Final context before returning:", context)  # Debugging final context
        return context

    def chat_interface(self):
        while True:
            user = input(">>> ")
            if user == "stop":
                break
            print(self.get_response(user))

    def summarize_context(self, context):
        if len(context) > 500:  # If context is long, summarize it
            # Use a transformer-based model for summarization (e.g., BART, T5)
            summary = self.summarizer(context, max_length=150, min_length=50, do_sample=False)
            return summary[0]['summary_text']
        else:
            # If the context is short, return it as is
            return context

if __name__ == "__main__":
    company = "CBRE"
    load_dotenv()
    
    # Initialize the model
    model = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=1024,
        top_k=50,
    )
    
    # Create an instance of the RAG class
    r = RAG(company, model)
    r.chat_interface()
