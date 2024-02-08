import argparse 
import os 
import json 
import chromadb
from llama_index.retrievers import VectorIndexRetriever 
from llama_index.vector_stores import ChromaVectorStore
from llama_index.readers.chroma import ChromaReader
from llama_index.vector_stores import SimpleVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SentenceSplitter 
from llama_index import StorageContext, load_index_from_storage, load_indices_from_storage
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

class LocalStorage():
    def __init__(self, engine_type):
        self.index_path = '/workspace/db/local'
        self.engine_type = engine_type 

    def load_storage(self):
        if self.engine_type == 'desc':
            self.vector_path = os.path.join(os.path.join(self.index_path, 'demo', 'desc', 'default__vector_store.json'))
            self.vectorstore = SimpleVectorStore.from_persist_path(self.vector_path)
        elif self.engine_type == 'features':
            self.vectorstore = SimpleVectorStore.from_persist_path(os.path.join(self.index_path, 'demo', 'features', 'default__vector_store.json'))
        elif self.engine_type == 'qualification':
            self.vectorstore = SimpleVectorStore.from_persist_path(os.path.join(self.index_path, 'demo', 'qualification', 'default__vector_store.json'))

    def load_index(self, embedding_model):
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=30)   # SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        self.embedding_service = ServiceContext.from_defaults(node_parser=parser, embed_model=self.embed_model, llm=None)
        
        if self.engine_type == 'desc':
            self.storage_context = StorageContext.from_defaults(persist_dir=os.path.join(self.index_path, 'demo', 'desc'))
            self.vector_idx = load_index_from_storage(self.storage_context, index_id='scrapping_desc', service_context=self.embedding_service)
        elif self.engine_type == 'features':
            self.storage_context = StorageContext.from_defaults(os.path.join(self.index_path, 'demo', 'features'))
            self.vector_idx = load_index_from_storage(self.storage_context, index_id='scrapping_features', service_context=self.embedding_service)
        elif self.engine_type == 'qualification':
            self.storage_context = StorageContext.from_defaults(os.path.join(self.index_path, 'demo', 'qualification'))
            self.vector_idx = load_index_from_storage(self.storage_context, index_id='scrapping_qualification', service_context=self.embedding_service)


class ChromaDB():
    def __init__(self, config):
        self.config = config 
        self.index_path = config['index_path']
        self.db_path = config['vectordb_path']
        self.engine_type = config['engine_type']   
        self.emb_model = HuggingFaceEmbedding(model_name=self.config['embedding_model'] )
        self.parser = SentenceSplitter(chunk_size=512, chunk_overlap=30)
        self.embedding_service = ServiceContext.from_defaults(node_parser=self.parser, embed_model=self.emb_model, llm=None)

    def connect(self):
        self.client = chromadb.PersistentClient(path=self.db_path)

    def set_node_parser(self, node_parser, chunk_size, chunk_overlap):
        '''
        node_parser 환경 설정 (default - chunk size 512, overlap 30)
        '''
        self.parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 

    def set_emb_model(self, emb_model):
        '''
        embedding model 변경 (default: kakaobank)
        '''
        self.emb_model = HuggingFaceEmbedding(model_name=emb_model)

    def set_service_context(self, node_parser, embed_model, llm=None):
        self.embedding_service = ServiceContext.from_defaults(node_parser, embed_model, llm)

    def get_collection(self):
        if self.engine_type == 'desc':   # 상품 설명
            collection = self.client.get_or_create_collection('desc')
        elif self.engine_type == 'features':   # 상품 특징 
            collection = self.client.get_or_create_collection('features')
        elif self.engine_type == 'qualification':   # 자격 요건 
            collection = self.client.get_or_create_collection('qualification')
        return collection    
    
    def get_vector_store(self, collection):
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return vector_store

    def get_vector_index(self, vector_store):       
        vector_idx = VectorStoreIndex.from_vector_store(vector_store, service_context=self.embedding_service)
        return vector_idx 

    def store_idx(self, save_path):
        pass 
