from llama_index import SimpleDirectoryReader, VectorStoreIndex

from flask import jsonify
from flask_cors import CORS
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    SQLDatabase,

)


import os
import pinecone
# find API key in console at https://app.pinecone.io/
os.environ['PINECONE_API_KEY'] = '69cce988-1524-4dd4-bcf1-8221a8b5a576'
# environment is found next to API key in the console
os.environ['PINECONE_ENVIRONMENT'] = 'gcp-starter'

# initialize connection to pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)
pinecone_index = pinecone.Index('quickstart')
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore

from llama_index.llms import GradientBaseModelLLM
from llama_index.finetuning.gradient.base import GradientFinetuneEngine

os.environ["GRADIENT_ACCESS_TOKEN"] = "y80nybVVMkKYuUTWnAjymEtqdhUU0YQN"
os.environ["GRADIENT_WORKSPACE_ID"] = "6c0bcdd2-cc0b-43cc-8bd3-61211649ebf6_workspace"

import locale
locale.getpreferredencoding = lambda: "UTF-8"

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import  GradientBaseModelLLM
from llama_index.program import LLMTextCompletionProgram
from llama_index.output_parsers import PydanticOutputParser


gradient_handler = LlamaDebugHandler()
gradient_callback = CallbackManager([gradient_handler])
base_model_slug = "llama2-7b-chat"
llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug,
    max_tokens=300,
    callback_manager=gradient_callback,
    is_chat_model=True,
)

from llama_index.llms import GradientBaseModelLLM
from llama_index.finetuning.gradient.base import GradientFinetuneEngine
os.environ["GRADIENT_ACCESS_TOKEN"] = "y80nybVVMkKYuUTWnAjymEtqdhUU0YQN"
os.environ["GRADIENT_WORKSPACE_ID"] = "6c0bcdd2-cc0b-43cc-8bd3-61211649ebf6_workspace"


from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import  GradientBaseModelLLM
from llama_index.program import LLMTextCompletionProgram
from llama_index.output_parsers import PydanticOutputParser


gradient_handler = LlamaDebugHandler()
gradient_callback = CallbackManager([gradient_handler])
base_model_slug = "llama2-7b-chat"
llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug,
    max_tokens=300,
    callback_manager=gradient_callback,
    is_chat_model=True,
)
#folder_name = "data/wms"

# Get the current directory
#current_directory = os.getcwd()

# Create a path for the new folder
#wikipedia_data_dir= folder_name

from llama_index import set_global_service_context

from llama_index.embeddings import GradientEmbedding

embed_model = GradientEmbedding(
    gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
    gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
    gradient_model_slug="bge-large",
)


service_context = ServiceContext.from_defaults(
    llm = llm,
    embed_model = embed_model,
    chunk_size=256,
)

set_global_service_context(service_context)



from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex



# connect and name the PineconeVectorStore
wms_vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="wikipedia_info"
)

# allow the PineconeVectorStore to be used as storage
storage_context = StorageContext.from_defaults(vector_store=wms_vector_store)


documents = SimpleDirectoryReader(wikipedia_data_dir).load_data()
# allow the creation of an Index
wms_vector_index = VectorStoreIndex.from_documents([],
                                       storage_context=storage_context)
query_engine = wms_vector_index.as_query_engine()



from flask import Flask
from pyngrok import ngrok
from flask import Flask, request,render_template
import requests

portno = 8000


app = Flask(__name__)
CORS(app)

#ngrok.set_auth_token('2ZGr4Gf5m7wNAUvDwHdgdSSZJPX_4mGbGzSCpfAFuc7fPpGGi')
#public_url = ngrok.connect(portno).public_url

wms_vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="wikipedia_info")

# Allow the PineconeVectorStore to be used as storage
storage_context = StorageContext.from_defaults(vector_store=wms_vector_store)

# Allow the creation of an Index

def download_wikipedia_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Save the content to a file
        with open(os.path.join(wikipedia_data_dir, 'wikipedia_content.txt'), 'w', encoding='utf-8') as file:
            file.write(response.text)

def load_data_into_vector_store():
    # Load data into PineconeVectorStore
    #documents = SimpleDirectoryReader(wikipedia_data_dir).load_data()
    wms_vector_index = VectorStoreIndex.from_documents([],
                                       storage_context=storage_context)

def train_query_engine_with_data():
    # Train the query engine with loaded data
    query_engine = wms_vector_index.as_query_engine()
@app.route("/")
def home():
  return render_template("chat.html")
@app.route('/prompt/<input_text>', methods=['GET'])
def get_response(input_text):
    # Call your query_engine function with the input_text
    bot_response = query_engine.query(input_text)

    # Prepare the response data
    response_data = {
        'messages': [
            {'content': input_text}
        ],
        'candidates': [
            {'content': bot_response.response}
        ]
    }

    # Return the response as JSON
    return response_data
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    bot_response = query_engine.query(input)

    return bot_response.response
@app.route('/train_query_engine', methods=['POST','GET'])
def train_query_engine():
    wikipedia_url = request.form.get('wikipedia_url')
    print(wikipedia_url)
    # Create the directory if it doesn't exist
    os.makedirs(wikipedia_data_dir, exist_ok=True)
    # Download Wikipedia content based on the provided URL
    download_wikipedia_content(wikipedia_url)
    load_data_into_vector_store()
    train_query_engine_with_data()
    # Return a response indicating that training is complete
    return jsonify({'status': 'Training complete'})


#print(f"to access go to {public_url}")
