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
    t = re.sub(r"[^a-zA-Z ]+", " ", t)
    t = re.sub(" +", " ", t)
    t = t.lower()
    return t


def normalize_vector(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


class SearchModel:
    def __init__(self):
        self.loader = DirectoryLoader("data/documents/", glob="*.pdf")
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=50,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.train()

    def train(self):
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
        cur.execute(f"SELECT * FROM pages")
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
        self.bm25 = BM25Okapi(tokenized_docs)
        self.split_documents = split_documents
        """try:
            db.delete_collection()
        except:
            pass
        db = Chroma.from_documents(split_documents, self.embeddings_model)
        self.retriever = db.as_retriever()"""

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with open(os.path.join("data/intent_vectors.pickle"), "wb") as f:
            pickle.dump(self.intent_vectors, f)
        with open(os.path.join("data/intent_labels.pickle"), "wb") as f:
            pickle.dump(self.intent_labels, f)

    def load(self):
        """Loads trained component"""
        try:
            with open(os.path.join("data/intent_vectors.pickle"), "rb") as f:
                self.intent_vectors = pickle.load(f)
            with open(os.path.join("data/intent_labels.pickle"), "rb") as f:
                self.intent_labels = pickle.load(f)

        except:
            self.intent_vectors = {}
            self.intent_labels = {}

    def query(self, question):
        """Queries the chatbot"""

        tokenized_query = self.splitter.tokenizer.tokenize(clean_text(question))
        return [
            [
                r.metadata["doc_id"],
                r.page_content,
                r.metadata["doc_real_id"],
                r.metadata["page"],
            ]
            for r in self.bm25.get_top_n(tokenized_query, self.split_documents, n=5)
        ]


# return [r.page_content for r in self.retriever.invoke(question)]
