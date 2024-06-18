import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from unidecode import unidecode
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import psycopg2
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
from unidecode import unidecode
import re
from postgres_secrets import *


def clean_text(t):
    t = unidecode(t)
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"[^a-zA-Z ]+", " ", t)
    t = re.sub(" +", " ", t)
    t = t.lower()
    return t


def normalize_vector(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


class ScoredBM25Okapi(BM25Okapi):
    def get_top_n_with_score(self, query, documents, n=5):
        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [(documents[i], scores[i] / len(query)) for i in top_n]


class SearchModel:
    def __init__(self):
        self.loader = DirectoryLoader("data/documents/", glob="*.pdf")
        model_name = "./data/model_data/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=50,
            model_name=model_name,
        )
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.load()

    def train(self, system_id):
        """Trains a chatbot based on intent_examples

        Args:
            intent_examples List<Tuple<String, String>>: List of Tuples (intent id, question)

        Returns:
            None
        """

        conn = psycopg2.connect(
            dbname=DBNAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
        )
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM pages WHERE system_id='{system_id}'")
        examples = cur.fetchall()
        cur.close()
        conn.close()

        documents = []
        for example in examples:
            doc = Document(
                page_content=example[1],
                metadata={
                    "source": "local",
                    "doc_id": example[0],
                    "doc_real_id": example[2],
                    "page": example[3],
                },
            )
            documents.append(doc)
        split_documents = self.splitter.split_documents(documents)
        tokenized_docs = []
        for split_document in split_documents:
            clean_content = clean_text(split_document.page_content)
            tokenized_doc = self.splitter.tokenizer.tokenize(clean_content)
            tokenized_docs.append(tokenized_doc)
        self.bm25_indices[system_id] = ScoredBM25Okapi(tokenized_docs)
        self.split_documents[system_id] = split_documents

        self.persist()
        """try:
            db.delete_collection()
        except:
            pass
        db = Chroma.from_documents(split_documents, self.embeddings_model)
        self.retriever = db.as_retriever()"""

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with open("bm25_indices.pickle", "wb") as f:
            pickle.dump(self.bm25_indices, f)
        with open("split_documents.pickle", "wb") as f:
            pickle.dump(self.split_documents, f)

    def load(self):
        """Loads trained component"""
        try:
            with open("bm25_indices.pickle", "rb") as f:
                self.bm25_indices = pickle.load(f)
            with open("split_documents.pickle", "rb") as f:
                self.split_documents = pickle.load(f)
            print("Loaded existing indices.")
        except:
            # self.bm25_indices = {}  # system_id: bm25
            # self.split_documents = {}
            self.reload()
            print("Started new indices.")

    def reload(self):
        for i in range(1, 6):
            self.train(str(i))

    def query(self, question, system_id):
        """Queries the chatbot"""

        tokenized_query = self.splitter.tokenizer.tokenize(clean_text(question))
        return_documents = [
            {
                "PageID": r[0].metadata["doc_id"],
                "Text": r[0].page_content,
                "DocumentID": r[0].metadata["doc_real_id"],
                "Page": r[0].metadata["page"],
                "Score": r[1],
            }
            for r in self.bm25_indices[system_id].get_top_n_with_score(
                tokenized_query, self.split_documents[system_id], n=5
            )
        ]
        return return_documents


# return [r.page_content for r in self.retriever.invoke(question)]
