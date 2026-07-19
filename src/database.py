"""
All interactions with our local vector database (ChromaDB) live here.
Responsibilities: initialize a persistent client, populate it from the
offline facts file, and query it for facts relevant to a sport.
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, SPORTS_FACTS_PATH

_embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client saving to disk."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection():
    """Fetches (or creates) the sports_history collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )


def setup_and_populate_db(json_file_path: str = SPORTS_FACTS_PATH, force: bool = False):
    """
    Reads the offline JSON facts, creates a collection, and populates it.
    Safe to call on every app startup -- it only inserts data once,
    unless force=True is passed (useful after editing sports_facts.json).
    """
    collection = get_collection()

    if collection.count() > 0 and not force:
        return collection

    if force and collection.count() > 0:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Fact data file not found at {json_file_path}")

    with open(json_file_path, "r") as f:
        facts_list = json.load(f)

    documents, metadata_list, ids = [], [], []
    for idx, item in enumerate(facts_list):
        documents.append(item["fact"])
        metadata_list.append({"sport": item["sport"]})
        ids.append(f"fact_{idx}")

    collection.add(documents=documents, metadatas=metadata_list, ids=ids)
    return collection


def query_historic_facts(sport: str, query_text: str, n_results: int = 3):
    """
    Queries ChromaDB for historic documents relating to a sport.
    Filters results to only the selected sport category via metadata.
    Returns a list of fact strings (possibly empty).
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # n_results can't exceed the number of matching docs in the filtered subset,
    # so cap it defensively against the collection size.
    safe_n = min(n_results, collection.count())

    results = collection.query(
        query_texts=[query_text],
        n_results=safe_n,
        where={"sport": sport},
    )
    return results.get("documents", [[]])[0]
