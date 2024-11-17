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
        self.llm = OpenAI(temperature=0.8, max_tokens=500)
        self.embeddings = OpenAIEmbeddings()
        self.pca = PCA(n_components=1536)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
        
        index_name = "company-key-data"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = pc.Index(index_name)
        self.retriever = self.index.query
        self.model = model
        self.company = company
        self.conversation_history = ""
    
    def update_history(self, question, answer):
        self.conversation_history += f"Representative: {question}\nAgent: {answer}\n"
        history_parts = self.conversation_history.split("\n")
        if len(history_parts) > 6:  # Limiting history to 3 exchanges (6 lines)
            self.conversation_history = "\n".join(history_parts[-6:])
    
    def get_response(self, input_query, user_prompt):
        # Perform a Pinecone query to retrieve the closest matches
        context = self.retrieve_context(user_prompt)
        print('Retrieved Context:', context)  # Debugging context
        
        # Build the entire prompt as a single string
        prompt = f"""
        You are an assistant helping a representative from {self.company} with questions about its sustainability practices.  
        Below is the context about {self.company}'s goals:

        {context}

        Focus solely on {self.company}’s goals, critically analyzing the provided context to identify missing metrics, improvement areas, or opportunities for progress. Include actionable suggestions and relevant statistics when applicable.  
        Respond clearly and concisely, using no more than 3 sentences. Avoid filler or speculation beyond the given context.

        """
        
        output = self.llm(prompt)
        print('=============================================')
        print(context)
        
        # Update history
        self.update_history(user_prompt, output)
        
        return output
    
    def retrieve_context(self, query):
        # Convert the query into an embedding vector
        query_vector = self.embeddings.embed_query(query)
        
        query_vector = query_vector.tolist()

        # Perform a Pinecone query to retrieve relevant documents
        results = self.retriever(
            vector=query_vector,  # The embedding vector of the query (as a list)
            top_k=5,  # Number of similar results to fetch
            include_values=True,  # Optionally include the vector values
            include_metadata=True  # Optionally include metadata associated with the results
        )

        # Concatenate all relevant results into a single string
        context = ""
        for result in results['matches']:
            description = result['metadata'].get('description', 'No description available')
            company = result['metadata'].get('company', 'No company available')
            tags = ", ".join(result['metadata'].get('tags', ['No tags available']))
            
            # Combine the extracted fields into a coherent context
            text = f"Description: {description}\nTags: {tags}\n"
            context += text + "\n"
        
        # Handle edge cases with no results
        if not context.strip():
            context = "No relevant context found for the given query."
        
        return context

    def chat_interface(self):
        while True:
            user = input(">>> ")
            if user == "stop":
                break
            response = self.get_response(user, user)
            print(response)



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
