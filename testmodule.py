# from store_embeddings import store_embeddings, get_chroma_collection, retrieve_relevant_chunks

# store_embeddings("test")

# collection = get_chroma_collection()
# print(collection)

# query = "What is the capital of France?"
# relevant_chunks = retrieve_relevant_chunks(query)
# print(relevant_chunks)


# chunks = split_document("Your document content here...")
# store_embeddings(chunks)
# Retrieve relevant chunks based on a query
# query = "Sample query text"
# relevant_chunks = retrieve_relevant_chunks(query)
# st.write("Relevant Chunks:", relevant_chunks)
import streamlit as st
from langchain_community.llms import Ollama
from storage import get_relevant_chunks
from sentence_transformers import SentenceTransformer

# st.session_state.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# relevant_chunks = get_relevant_chunks("What is Ratio of Digits in Hostname?")
# print(relevant_chunks)
ollama = Ollama(
    base_url="http://localhost:11434",
    model="llama3.2Q80:latest"
)
test_prompt = "tách từ trong xử lí ngôn ngữ tự nhiên là gì? hãy trả lời chi tiết nhất có thể"
response = ollama.generate(
        prompts=[test_prompt],
        generation_config={
            'max_tokens': 4096,        # Increase max output length
            'temperature': 0.7,        # Add some randomness
            'top_p': 0.9,             # Nucleus sampling
            'num_predict': 1024,       # Number of tokens to predict
            'stop': ['\n\n\n'],       # Stop sequence
            'repeat_penalty': 1.1,     # Reduce repetition
        }
    )
# response = ollama.invoke(
#         model="llama3.2Q80:latest",
#         input=test_prompt,
#         options={
#             'num_predict': 1024,
#             'temperature': 0.7,
#             'top_p': 0.9,
#         }
#     )
# #response = ollama.invoke(input=test_prompt)
#print(response)
print(response.generations[0][0].text)
