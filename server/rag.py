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



class RAG():
    def __init__(self, company, model):
        # Load environment variables
        load_dotenv()

        # Set up embeddings using HuggingFace model
        self.embeddings = OpenAIEmbeddings()
        
        # Pinecone setup
        # Create a Pinecone instance 
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
        
        # Ensure the index exists or create it
        index_name = "company-key-data"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # The dimensionality of the embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Modify this to fit your region and cloud
            )
        
        # Connect to the index
        self.index = pc.Index(index_name)

        self.retriever = self.index.query  # Use the query method for retrieval

        self.model = model
        self.llm = OpenAI(temperature=0.8, max_tokens=500)
        self.company = company

        # Custom prompt template tailored for CBRE sustainability
        template = """
        <s>[INST] Focus solely on CBRE's goals, critically analyzing the provided context to identify gaps, missed opportunities, or areas where progress is lagging. Highlight specific metrics or initiatives that need improvement and suggest actionable steps to accelerate progress. Include relevant statistics when available and avoid excessive praise.  
        Respond clearly and concisely, using no more than 3 sentences. Avoid filler or speculation beyond the given context. Do not number your sentences. Avoid repeating the same concept in different phrasing. If asked a follow-up question ("e.g., Tell me more, or What's next"), avoid repeating the previous answer and provide fresh insights. 
        Do NOT include prefixes such as 'Agent', 'Response', 'Answer', or any other prefix in your response. You should respond like a human. Do NOT praise CBRE excessively. Focus on constructive critiscm. Limit your response to three sentences[/INST]


        Context: {context}

        Conversation History:
        {history}

        Representative: {question} [/INST]
        """

        self.prompt = ChatPromptTemplate.from_template(template)

        # Initialize conversation history
        self.conversation_history = ""
    
    def update_history(self, question, answer):
        self.conversation_history += f"Question: {question}\nAnswer: {answer}\n"

    def get_response(self, input_query, user_prompt):
        # Perform a Pinecone query with the correct top_k parameter
        results = self.retriever(
            vector=input_query,  # Pass the query vector (ensure input_query is the vector)
            top_k=5,  # Specify the number of similar results to retrieve
            include_values=True,  # Optionally include vector values
            include_metadata=True  # Optionally include metadata
        )
        
        # Extract and format the context from Pinecone results using metadata fields
        context = []
        for match in results.get('matches', []):  # Safeguard if 'matches' is not present
            metadata = match.get('metadata', {})
            
            # Get relevant metadata fields
            description = metadata.get('description', 'No description available.')
            value = metadata.get('value', 'N/A')
            tags = metadata.get('tags', [])
            
            # Format the context for this particular match
            match_context = f"{description}. Value: {value}. Tags: {', '.join(tags)}."
            context.append(match_context)
        
        # Combine context into a single string or provide a fallback message
        context_str = " ".join(context) if context else "No relevant context found."
        
        # Construct a single prompt with all necessary information
        prompt = (
            f"Context: {context_str}\n\n"
            f"Conversation History: {self.conversation_history}\n\n"
            f"Company: {self.company}\n\n"
            f"Question: {user_prompt}\n\n"
            "Based on the above information, provide a response."
        )
        
        # Pass the prompt to the model and get the output
        try:
            output = self.llm(prompt)
        except Exception as e:
            output = f"An error occurred while generating the response: {e}"
        
        # Update the conversation history
        self.update_history(user_prompt, output)
        
        return output  # Return the generated response



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