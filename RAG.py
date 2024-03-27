"""
    Import necessary libraries
"""

import os
import pandas as pd
import numpy as np
import json
import random
import chromadb
from chromadb import Settings
import chromadb.utils.embedding_functions as embedding_functions
from collections import Counter
import re
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

"""
    Load OpenAI API key from environment variable
"""

oai_api_key = os.getenv("OPENAI_API_KEY")
if oai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

"""
    Setup paths, define embedding function for queries and get chromadb collection
"""

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'chromaDB')
client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, anonymized_telemetry=False))
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=oai_api_key, model_name="text-embedding-3-large")
collection = client.get_collection(name="data", embedding_function=openai_ef)

"""
    Shuffle dataset to ensure randomness of questions

    Define checkpoint/output path for predictions by RAG system
"""

SHUFFLED_DATASET_PATH = os.path.join(DIR, 'data/shuffled_squad_test.parquet')
CHECKPOINT_PATH = 'data/predictions.json'

"""
    Setup RAG-Chain:
        - prompt template can be found here: https://smith.langchain.com/hub/nneubacher/rag-squad?organizationId=a8b46d86-6766-5a35-ad40-9504766b9f82
        - define LLM model and temperate used for answer generation
"""

prompt = hub.pull("nneubacher/rag-squad")
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.1)
rag_chain = prompt | llm | StrOutputParser()

"""
    Function for shuffling dataset:
        - pass if shuffled dataset already exist 
"""

def shuffle_dataset_once(input_path, output_path):
    """Shuffles the dataset if not already shuffled and saved."""
    if not os.path.exists(output_path):
        df = pd.read_parquet(input_path)
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_shuffled.to_parquet(output_path)
        print("Dataset shuffled and saved.")
    else:
        print("Shuffled dataset already exists.")

"""
    Define loader for questions:
        - iterates over rows of shuffled dataset
        - retrieves question and ground truth answer
"""

class SQuADLoaderQnA:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> dict:
        df = pd.read_parquet(self.file_path)
        documents = {}
        for index, row in df.iterrows():
            id = row['id']
            question = row['question']
            context = row['context']
            answers_text = row['answers']['text'].tolist() if 'answers' in row and row['answers'] and 'text' in row['answers'] else ["No Answer"]
            documents[id] = {'content': {'question': question, 'answers': answers_text, 'context': context}}
        return documents
    

"""
    Function for querying the collection:
        - uses chromadb's .query() function to return most similar documents
            - embedds question as vector with same dimensionality as collection (3072)
            - uses cosine similarity to find closest match
            - returns top 2 matches and concatenates the 2 contexts
"""

def query_collection(collection, question, top_k=2):
    contexts = []

    query_result = collection.query(query_texts=[question], n_results=top_k, include=["documents"]).get('documents')
    if query_result:
        for content_list in query_result:
            combined_content = " ".join(content_list)
            contexts.append({'context': combined_content})
    return contexts    

"""
    Utility function for validating predicted answer with ground truth answer:
        - returns all sequences of letters, digits and underscores in lowecase
"""

def preprocess(text):
    return re.findall(r'\w+', text.lower())

"""
    Function to calculate the Jaccard similarity of predicted answer and ground truth answer:
        - determines the similarity of two sets of tokens
            - equals number of overlapping tokens in both sets divided by number of tokens in either sets
            - returns value between 0 and 1, 1 if the sets share the exact same tokens
"""

def calculate_jaccard_similarity(predicted_answer, true_answers):
    predicted_tokens = set(preprocess(predicted_answer))
    best_similarity = 0
    for true_answer in true_answers:
        true_tokens = set(preprocess(true_answer))
        intersection = predicted_tokens & true_tokens
        union = predicted_tokens | true_tokens
        similarity = len(intersection) / len(union)
        best_similarity = max(best_similarity, similarity)
    return best_similarity

"""
    Computes the cosine similarity between the predicted answer and each true answer,
    returning the highest score.
"""

def calculate_cosine_similarity(predicted_answer, true_answers):
    # Start with the default TfidfVectorizer configuration
    tfidf_vectorizer = TfidfVectorizer()
    best_similarity = 0.0

    documents = [predicted_answer] + true_answers
    try:
        # First attempt with default settings
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        cosine_similarities = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        best_similarity = max(cosine_similarities[0])
    except ValueError as e:
        # Fallback to a more permissive TfidfVectorizer configuration if the first attempt fails
        tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None)
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            # Check if the second attempt was successful
            if tfidf_matrix.shape[1] > 0:
                cosine_similarities = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
                best_similarity = max(cosine_similarities[0])
        except ValueError:
            # If vectorization fails again, log the error or handle it as needed
            print(f"TF-IDF vectorization failed again: {e}. Predicted answer: '{predicted_answer}', True answer: '{true_answers}'")

    return best_similarity

"""
    Function for evaluation of similarity:
        - evaluates predicted answer similar to ground truth answer using Jaccard similarity
            - preliminary threshold arbitrarily set to 0.4 
"""

def evaluate_answers_jaccard(predicted_answer, true_answers, threshold=1.0):
    similarity = calculate_jaccard_similarity(predicted_answer, true_answers)
    return similarity >= threshold, similarity

def evaluate_answers_cosine(predicted_answer, true_answers, threshold=0.4):
    similarity = calculate_cosine_similarity(predicted_answer, true_answers)
    return similarity >= threshold, similarity

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

"""
    Function for saving evaluation:
        - saves question, predicted answer, ground truth answer and similarity score => 'data'
        - updates number of correct and total predictions
"""

def save_checkpoint(data, current_index, correct_predictions_jaccard, correct_predictions_cosine, total_predictions, correct_contexts, total_questions, file_path=CHECKPOINT_PATH):
    checkpoint_data = {
        "data": data,
        "current_index": current_index,
        "correct_predictions_jaccard": correct_predictions_jaccard,
        "correct_predictions_cosine": correct_predictions_cosine,
        "total_predictions": total_predictions,
        "correct_contexts": correct_contexts,
        "total_questions": total_questions
    }
    with NamedTemporaryFile('w', delete=False) as tf:
        json.dump(checkpoint_data, tf, cls=NumpyEncoder)
        temp_name = tf.name
    os.replace(temp_name, file_path)

"""
    Function for crash recovery:
        - if evaluation run fails, following run will load prediction checkpoint and ensure no redundant evalutation
"""

def load_checkpoint(file_path=CHECKPOINT_PATH):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"data": [], "current_index": 0, "correct_predictions_jaccard": 0, "correct_predictions_cosine": 0, "total_predictions": 0, "correct_contexts": 0, "total_questions": 0}

"""
    Function to invoke RAG chain:
        - passes question and concatenated contexts into query template
        - returns predicted answer from LLM
"""

def get_answer_with_rag_chain(rag_chain, question, contexts):
    # Combine the contexts into a single string
    combined_context = " ".join(c['context'] for c in contexts)
    
    # Construct the input for the RAG chain
    rag_input = {"context": combined_context, "question": question}
    
    # Invoke the RAG chain with the constructed input
    predicted_answer = rag_chain.invoke(rag_input)
    
    return predicted_answer

"""
    Main function for evaluation:
        - shuffles dataset if not done already
        - loads dataset
        - checks for existing prediction checkpoint
        - iterates over (remaining) questions (for 10000 questions)
            - determines relevant contexts
            - invokes RAG chain and retrieves predicted answer
            - evaluates answer using Jaccard similarity
            - saves predictions and evaluates preliminary accuracy (based on the threshold of 0.4)
"""

def main_evaluation(file_path=SHUFFLED_DATASET_PATH, max_evaluations=10000):

    shuffle_dataset_once('data/squad_test.parquet', file_path)
    loader = SQuADLoaderQnA(file_path)
    squad_qa = loader.load()
    
    checkpoint = load_checkpoint()
    predictions_details = checkpoint.get('data', [])
    current_index = checkpoint.get('current_index', 0)
    correct_predictions_jaccard = checkpoint.get('correct_predictions_jaccard', 0)
    correct_predictions_cosine = checkpoint.get('correct_predictions_cosine', 0)
    total_predictions = checkpoint.get('total_predictions', 0)
    correct_contexts = checkpoint.get('correct_contexts', 0)
    total_questions = checkpoint.get('total_questions', 0)


    print(f"Resuming from document index {current_index + 1} of {len(squad_qa)}")

    max_index = min(current_index + (10000 - total_predictions), len(squad_qa))

    for doc_id, doc in tqdm(list(squad_qa.items())[current_index:max_index], initial=current_index, total=max_index, desc="Evaluating"):
        question = doc['content']['question']
        true_context = doc['content']['context']
        true_answers = doc['content']['answers']
        
        contexts = query_collection(collection, question)

        context_found = any(true_context in c['context'] for c in contexts)

        if context_found:
            correct_contexts += 1
        total_questions += 1

        predicted_answer = get_answer_with_rag_chain(rag_chain, question, contexts)
        
        is_similar_jaccard, similarity_score_jaccard = evaluate_answers_jaccard(predicted_answer, true_answers)

        is_similar_cosine, similarity_score_cosine = evaluate_answers_cosine(predicted_answer, true_answers)
        
        predictions_details.append({
            "doc_id": doc_id,
            "question": question,
            "predicted_answer": predicted_answer,
            "true_answers": true_answers,
            "is_similar_jaccard": is_similar_jaccard,
            "similarity_score_jaccard": similarity_score_jaccard,
            "is_similar_cosine": is_similar_cosine,
            "similarity_score_cosine": similarity_score_cosine,
            "context_found": context_found
        })
        if is_similar_jaccard:
            correct_predictions_jaccard += 1
        if is_similar_cosine:
            correct_predictions_cosine += 1
        total_predictions += 1

        current_index += 1
        save_checkpoint(data=predictions_details, current_index=current_index, correct_predictions_jaccard=correct_predictions_jaccard, correct_predictions_cosine=correct_predictions_cosine, total_predictions=total_predictions, correct_contexts=correct_contexts, total_questions=total_questions, file_path='checkpoint.json')
        
        if total_predictions >= 10000:
            break

    if total_predictions > 0:
        accuracy_jaccard = correct_predictions_jaccard / total_predictions
        accuracy_cosine = correct_predictions_cosine / total_predictions
        recall = correct_contexts / total_questions
        print(f"Accuracy with Jaccard: {accuracy_jaccard:.2f}")
        print(f"Accuracy with Cosine: {accuracy_cosine:.2f}")
        print(f"Recall: {recall:.2f}")
    else:
        print("No predictions were made.")

if __name__ == "__main__":
    main_evaluation()