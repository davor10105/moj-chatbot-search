import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from unidecode import unidecode
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


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

        documents = self.loader.load()
        split_documents = self.splitter.split_documents(documents)
        db = Chroma.from_documents(split_documents, self.embeddings_model)
        self.retriever = db.as_retriever()

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

        return [r.page_content for r in self.retriever.invoke(question)]
