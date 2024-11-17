from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_together import Together
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

class RAG():
    def __init__(self, company, model):
        # Load environment variables
        load_dotenv()

        # Set up embeddings using HuggingFace model
        self.embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5")
        
        # Pinecone setup
        # Create a Pinecone instance 
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
        
        # Ensure the index exists or create it
        index_name = "company-key-data"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,  # The dimensionality of the embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Modify this to fit your region and cloud
            )
        
        # Connect to the index
        self.index = pc.Index(index_name)

        self.retriever = self.index.query  # Use the query method for retrieval

        self.model = model
        self.company = company

        # Custom prompt template tailored for CBRE sustainability
        template = """
        <s>[INST] You are an agent speaking with a representative from CBRE.  
        Your goal is to assist the representative with questions regarding CBRE's sustainability practices.  
        You are provided with context on CBRE’s sustainability goals. Focus only on CBRE’s practices and avoid discussing other companies. Be critical by identifying any missing metrics or areas for improvement, and provide specific, actionable suggestions.  
        Answer the question with a concise, focused response. Split your response into clear, logical paragraphs to improve readability. Use only the provided context to inform your answer. Don't give more than 3 sentences before a line break [/INST]


        Context: {context}

        Conversation History:
        {history}

        Representative: {question} [/INST]

        """
        self.prompt = ChatPromptTemplate.from_template(template)

        # Initialize conversation history
        self.conversation_history = ""
    
    def update_history(self, question, answer):
        self.conversation_history += f"Representative: {question}\nAgent: {answer}\n"

    def get_response(self, input_query, user_prompt):
        # Perform a Pinecone query with the correct top_k parameter
        results = self.retriever(
            vector=input_query,  # Pass the query vector (ensure input_query is the vector)
            top_k=5,  # Specify the number of similar results to retrieve
            include_values=True,  # Optionally include vector values
            include_metadata=True  # Optionally include metadata
        )
        
    
        
        # Prepare chain input as a dictionary
        chain_input = {
            "history": lambda x: self.conversation_history,  # Assuming conversation_history is a string or similar format
            "company": lambda x: self.company,  # CBRE, as per your requirement
            "question": RunnablePassthrough()  # Original user question
        }
        
        # Define the chain with structured input
        chain = (
            chain_input  # Pass the structured input
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        
        # Invoke the chain and get the response
        output = chain.invoke(chain_input)  # Pass chain_input, which is a dictionary
        
        # Update the conversation history
        self.update_history(user_prompt, output)
        
        return output  # Only return the final output


    def chat_interface(self):
        while True:
            user = input(">>> ")
            if user == "stop":
                break
            print(self.get_response(user))

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