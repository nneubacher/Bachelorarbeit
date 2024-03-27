import os
import pandas as pd
from tqdm import tqdm
import subprocess
import signal
import time
import sys
import chromadb
from chromadb import Settings
import chromadb.utils.embedding_functions as embedding_functions

class SQuADLoaderV1Parquet:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> dict:
        df = pd.read_parquet(self.file_path)
        documents = {}
        for index, row in df.iterrows():
            id = row['id']
            question = row['question']
            answers = row['answers'] if 'answers' in row and row['answers'] else [{"text": "No Answer", "answer_start": 0}]
            context = row['context']
            title = row['title'] if 'title' in row else "No Title"  
            documents[id] = {'content': {'context': context, 'question': question, 'answers': answers, 'title': title}, 'metadata': {'context': context}}
        return documents

def initialize_client():
    DIR = os.path.dirname(os.path.abspath("__file__"))
    DB_PATH = os.path.join(DIR, 'testDB')
    return chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, anonymized_telemetry=False))

def create_embedding_function(api_key):
    return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-large")

def setup_collection(client, embedding_function):
    return client.get_or_create_collection(name="500", metadata={"hnsw:space": "cosine"}, embedding_function=embedding_function)

def load_counter():
    try:
        with open('last_counter.txt', 'r') as file:
            return int(file.read().strip())
    except (FileNotFoundError, ValueError):
        return 1

def save_counter(counter):
    with open('last_counter.txt', 'w') as file:
        file.write(str(counter))

def main():
    
    mitm_command = ['mitmdump', '-s', os.path.join(os.path.dirname(os.path.abspath("__file__")), 'logger.py')]
    if sys.platform == "win32":
        mitm_process = subprocess.Popen(mitm_command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, shell=True)
    else:
        mitm_process = subprocess.Popen(mitm_command, preexec_fn=os.setsid)
    try:

        os.environ["HTTP_PROXY"] = "http://localhost:8080"
        os.environ["HTTPS_PROXY"] = "http://localhost:8080"

        time.sleep(5)

        oai_api_key = os.getenv("OPENAI_API_KEY")
        if oai_api_key is None:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        client = initialize_client()
        embedding_function = create_embedding_function(oai_api_key)
        collection = setup_collection(client, embedding_function)

        squad_loader = SQuADLoaderV1Parquet(os.path.join(os.path.dirname(os.path.abspath("__file__")), 'data/squad_test_10.parquet'))
        squad_documents = squad_loader.load()

        unique_contexts = set([doc['content']['context'] for doc in squad_documents.values()])
        counter = load_counter()
        unique_contexts_list = list(unique_contexts)[counter-1:]
        max_id_length = len(str(len(unique_contexts)))

        for context in tqdm(unique_contexts_list, desc="Embedding", unit="context"):
            try:
                doc_metadata = {'context': context}
                new_id = str(counter).zfill(max_id_length)
                collection.add(documents=[context], metadatas=[doc_metadata], ids=[new_id])
                counter += 1
                save_counter(counter)
            except Exception as e:
                print(f"An error occurred while embedding context {counter}: {e}")

        print(f"Total embedded contexts: {counter}")
        print("Embedding done")

    finally:
        if sys.platform == "win32":
            mitm_process.terminate()
        else:
            os.killpg(os.getpgid(mitm_process.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()
