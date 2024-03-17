
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.storage.storage_context import StorageContext


from llama_index.embeddings import resolve_embed_model
from llama_index.llms import HuggingFaceLLM
import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms import OpenAI

from tqdm import tqdm
import os
import nest_asyncio
nest_asyncio.apply()
from transformers import GenerationConfig, BitsAndBytesConfig
import torch


def local_llm_model(MODEL_NAME, embed_name, type_ = "local", token_counter = False):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    if type_ == 'local':
        llm = HuggingFaceLLM(
            context_window=3900,
            max_new_tokens=512,
            tokenizer_name=MODEL_NAME,
            model_name=MODEL_NAME,
            device_map="auto",
            tokenizer_kwargs={
                "max_length": 4096,
                "token" : "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",
                # 'trust_remote_code' : True # qwen
                },
            model_kwargs={
                "token" : "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",
                "quantization_config": quantization_config, # not solar
                # 'trust_remote_code' : True # qwen,
                # 'torch_dtype' : torch.float16, # solar
                # 'low_cpu_mem_usage' : True,
                },
            generate_kwargs={
                "temperature": 0,
                "do_sample": False,
                },
        )

    elif type_ == 'openai':
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    else:
        ValueError("type_ must be either 'local' or 'openai'")

    embed_model = resolve_embed_model(f"local:{embed_name}")

    
    if token_counter :
        callback_manager = CallbackManager([
            TokenCountingHandler(
            tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002").encode,
            verbose = True
            )
        ])
    else:
        callback_manager = CallbackManager([])
    
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm = llm,
        callback_manager=callback_manager
    )
    
    return service_context, embed_model, llm



def load_db(embed_name, embed_size):
    
        
    # load vector store

    collection_name = os.path.basename(embed_name)
    # embed_size = 'h'

    import chromadb
    from llama_index.vector_stores import ChromaVectorStore
    # make chroma vectorstore
    # chroma_clint = chromadb.EphemeralClient()
    db = chromadb.PersistentClient(path=f'./Papers/chroma/{collection_name}_{embed_size}')
    chroma_collection =db.get_or_create_collection(f"{collection_name}_{embed_size}")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(persist_dir=f'./Papers/chroma/{collection_name}_{embed_size}')

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        # service_context=service_context,
    )
    
    return index, storage_context


from llama_index.retrievers import QueryFusionRetriever
from llama_index.retrievers import AutoMergingRetriever
from typing import List
from llama_index.schema import NodeWithScore, QueryBundle
from enum import Enum
class FUSION_MODES(str, Enum):
    """Enum for different fusion modes."""

    RECIPROCAL_RANK = "reciprocal_rerank"  # apply reciprocal rank fusion
    SIMPLE = "simple"  # simple re-ordering of results based on original scores
    
def _retrieve_union_scholar(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    if self.num_queries > 1:
        self.queries = self._get_queries(query_bundle.query_str)
    else:
        self.queries = [query_bundle.query_str]
    # print(queries)
    
    if self.use_async:
        results = self._run_nested_async_queries(self.queries)
    else:
        results = self._run_sync_queries(self.queries)
    # print(results)

    if self.mode == FUSION_MODES.RECIPROCAL_RANK:
        res = self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        
        return res
    elif self.mode == FUSION_MODES.SIMPLE:
        return self._simple_fusion(results)[: self.similarity_top_k]
    else:
        raise ValueError(f"Invalid fusion mode: {self.mode}")

def retriever_engine(
    top_k,
    retrieve_mode,
    index,
    storage_context,
    QUERY_GEN_PROMPT = None,
    query_fusion = True):
    
    if QUERY_GEN_PROMPT is None:
        QUERY_GEN_PROMPT = (
        "You are a thesis search assistant."
        "A user has a question about the content of your paper.A user has a question about the content of your paper."
        "Create {num_queries} search queries, one on each line, to help users find the information they need."
        "Must be related to the input query."
        "Query : {query}\n"
        "Queries : \n"
        )
    
    
    if retrieve_mode == 'window':
        from llama_index.retrievers import VectorIndexRetriever
        retriever = VectorIndexRetriever(
            index = index,
            similarity_top_k = top_k
        )
    elif retrieve_mode == 'hierarchical':
        ## retriever
        retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=top_k), 
            storage_context=storage_context,
            simple_ratio_thresh = 0.3
        )

    if query_fusion:
        final_retriever = QueryFusionRetriever(
            [retriever],#, bm25_retriever],
            # index.as_retriever(similarity_top_k=top_k),
            similarity_top_k=top_k-20,
            num_queries=3,  # set this to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=QUERY_GEN_PROMPT,  # we could override the query generation prompt here
        )
        
        # override the reciprocal rerank fusion method
        QueryFusionRetriever._retrieve = _retrieve_union_scholar
        
    else:
        final_retriever = retriever
        
    return final_retriever


def query_engine(
    retriever,
    query,
    node_postprocessors,
    service_context,
    DEFAULT_TEXT_QA_PROMPT_TMPL=None,
    DEFAULT_TREE_SUMMARIZE_TMPL=None,
    FINAL_QA_PROMPT_TMPL=None
    ):

    from collections import defaultdict
    from llama_index.prompts import PromptTemplate
    
    nodes = retriever.retrieve(query)

    # post process
    for postprocessor in node_postprocessors:
        nodes = postprocessor.postprocess_nodes(nodes, QueryBundle(query_str=query))
    
    res = ''
    num_content_dict = defaultdict(str)

    for idx, node in enumerate(nodes):
        partial_res = service_context.llm_predictor.predict(
            PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL).partial_format(query_str=query),
            context_str = node.text
            )
        # print(partial_res)
        res += '\n\n'
        res += f'## {idx}. Title : ' + node.metadata['file_name'] + '\n'
        res += partial_res
        num_content_dict[idx] = node #(node.metadata['file_name'], node.text)
        
    response = service_context.llm_predictor.predict(
        PromptTemplate(DEFAULT_TREE_SUMMARIZE_TMPL).partial_format(query_str=query),
        context_str = res
        )


    import re
    final_context = ''
    try:
        # final_list += eval(response)
        for i in eval(response):
            final_context += '\n'
            final_context += num_content_dict[i].text
    except:
        response = re.search("\[([^\]]*)\]", response).group(0)
        for i in eval(response):
            final_context += '\n'
            final_context += num_content_dict[i].text

    if final_context == '':
        final_response = 'No context provided'
            
    else:
        final_response = service_context.llm_predictor.predict(
            PromptTemplate(FINAL_QA_PROMPT_TMPL).partial_format(query_str=query),
            context_str = final_context
            )
        
    return final_response, response, num_content_dict, nodes