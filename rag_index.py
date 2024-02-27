# -*- coding: cp949 -*- 

import torch
import os 
import argparse 
import time 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig
from llama_index import QueryBundle 
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import RetrieverQueryEngine 
from llama_index.postprocessor import SimilarityPostprocessor 
from llama_index.postprocessor import SentenceTransformerRerank 
from llama_index.postprocessor import KeywordNodePostprocessor 
from llama_index.postprocessor import SimilarityPostprocessor, CohereRerank
from llama_index.schema import Node, NodeWithScore 
from llama_index.retrievers import VectorIndexRetriever 
from src.vector_store import LocalStorage


def get_info(node):
    '''
    id, text, score  
    '''
    node_info = dict() 
    node_info['id'] = vector_dict['metadata_dict'][node.node_id]['document_id']
    node_info['name'] = vector_dict['metadata_dict'][node.node_id]['name']
    node_info['text'] = node.text 
    node_info['score'] = node.score 
    return node_info  

def main(cli_argse): 
    global vector_dict 

    # Retrieve 
    storage = LocalStorage(cli_argse.engine_type)
    storage.load_storage()
    vector_store = storage.vectorstore
    vector_dict = vector_store.to_dict()
    storage.load_index(cli_argse.emb_model)
    vector_idx = storage.vector_idx 

    retriever = VectorIndexRetriever(index=vector_idx, service_context=storage.embedding_service, similarity_top_k=10, verbose=True)
    query_bundle = QueryBundle(cli_argse.query)
    start = time.time() 
    retrieved_nodes = retriever.retrieve(query_bundle)
    print(f'관련 노드 추출에 걸린 시간: {time.time() - start}')

    ## post process
    similarity_cutoff = 0.3 
    reranker_top_n = 1 
    with_reranker = True 

    node_postprocessors = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    processed_nodes = node_postprocessors.postprocess_nodes(retrieved_nodes)
    print(f'관련 노드 추출 + 유사도 필터링에 걸린 시간: {time.time() - start}')

    if with_reranker:   
        reranker = SentenceTransformerRerank(
            model='bongsoo/albert-small-kor-cross-encoder-v1',
            top_n=reranker_top_n,
        )
        reranked_nodes = reranker.postprocess_nodes(
            processed_nodes, query_bundle
        )
   
    print(f'관련 노드 추출 + 유사도 필터링 + 재순위화에 걸린 시간: {time.time() - start}')
    node_info = get_info(reranked_nodes[0])
    
    # Augment 
    context = f"'{node_info['name']}'은 {node_info['text']} 관련 상품입니다."
    prompt_template = (
    f"""<s>[INST] 질문: {cli_argse.query} \n
    관련 정보: {context} \n 
    관련 정보를 바탕으로 질문에 답해줘 [/INST] \n"""
    )

    # Generation
    model = AutoModelForCausalLM.from_pretrained(cli_argse.gen_model, torch_dtype=torch.float16, low_cpu_mem_usage=True) # , device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(cli_argse.gen_model)
    g_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model.to(g_device)

    # streamer = TextStreamer(tokenizer)
    generation_config = GenerationConfig(
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
        max_new_tokens=512,
    )

    gened = model.generate(
    **tokenizer(
        prompt_template,
        return_tensors='pt',
        return_token_type_ids=False
    ).to('cuda'),
    generation_config=generation_config,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    # streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])
    start_tag = f"[/INST]"
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()

    print(result_str)


if __name__ == '__main__':
    global cli_argse

    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--emb_model", type=str, default='kakaobank/kf-deberta-base')
    cli_parser.add_argument("--gen_model", type=str, default='davidkim205/komt-mistral-7b-v1')
    cli_parser.add_argument("--engine_type", type=str, default='desc', help='can query about description, features, qualification')
    cli_parser.add_argument("--query", type=str)
    cli_argse = cli_parser.parse_args()
    main(cli_argse)