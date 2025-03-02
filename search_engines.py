import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"

from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import TextNode

import tiktoken

Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
TOKENS_PER_DOCUMENT = 8000 # limit is actually 10K tokens, but I put a lower limit as we do some other calls as well


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class DocSearch:
    
    def __init__(self, text_storage_dir) -> None:
                
        all_nodes = self.chunk_all_files(text_storage_dir)
        
        self.query_engine = self.initialize_vector_store(all_nodes)
        self.bm25_retriever = self.initialize_BM25(all_nodes)

    #### GENERAL CHUNKING
    def chunk_all_files(self, text_storage_dir: str) -> list[TextNode]:

        documents = SimpleDirectoryReader(input_dir = text_storage_dir, recursive=True, required_exts=[".txt"]).load_data() # "./data/AzulSa/txt"

        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        return nodes



    #### EMBEDDING VECTOR SEARCH
    def initialize_vector_store(self, nodes: list[TextNode]) -> VectorStoreIndex:
        vector_index = VectorStoreIndex(nodes)
        
        # Create a retriever to search nodes
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=100)
        # Create a query engine
        query_engine = RetrieverQueryEngine(retriever=vector_retriever)
        return query_engine


    def hybrid_search(self, search_term: str, target_files: list[str], number_of_hits: int, weight_vector: float = 0.5, weight_BM25: float = 0.5, min_hit_score = None) -> list[TextNode]:
        
        ##################################################################
        ############## Perform search with each search algo ##############
        ##################################################################
        
        #### Vector search 
        results = self.query_engine.query(search_term)

        filtered_vector_nodes = [
            node for node in results.source_nodes
            if node.metadata['file_path'].split("/")[-1] in target_files # remove files not in scope
        ]
        
        #### BM25    
        retrieved_BM25_nodes = self.bm25_retriever.retrieve(
            search_term
        )
        
        filtered_BM25_nodes = [
            node for node in retrieved_BM25_nodes
            if node.metadata['file_path'].split("/")[-1] in target_files
        ]
        

        ################################################################
        ############ Normalize result from each search algo ############
        ################################################################

        #### BM25
        max_BM25_score = -float("inf")
        for node in filtered_BM25_nodes: # Get max score
            if node.score > max_BM25_score:
                max_BM25_score = node.score
        
        for node in filtered_BM25_nodes: # re-weight scores to be 0 to 1
            node.score = node.score/max_BM25_score
        
        #### Vector search
        max_vector_score = -float("inf")
        for node in filtered_vector_nodes: # Get max score
            if node.score > max_vector_score:
                max_vector_score = node.score
        
        for node in filtered_vector_nodes: # re-weight scores to be 0 to 1
            node.score = node.score/max_BM25_score
            
        
        ####################################################
        ############## Combine search results ##############
        ####################################################
        
        # First, build a new list with weighted scores from the vector results.
        weighted_vector_nodes = []
        for node in filtered_vector_nodes:
            # Multiply the vector search score by its weight
            weighted_score = weight_vector * node.score
            node.metadata["weighted_score"] = weighted_score
            weighted_vector_nodes.append(node)

        # Next, do the same for the BM25 results.
        weighted_BM25_nodes = []
        for node in filtered_BM25_nodes:
            weighted_score = weight_BM25 * node.score
            node.metadata["weighted_score"] = weighted_score
            weighted_BM25_nodes.append(node)
        
        combined_nodes = weighted_vector_nodes + weighted_BM25_nodes

        # Remove duplicates, where both search engines recommended the same node
        merged_nodes = {}
        for node in combined_nodes:
            # Here we use the file name and the first 30 characters of the text.
            node_key = f"{node.metadata.get('file_path','')}_{node.text[:30]}"
            if node_key in merged_nodes:
                # Sum the weighted scores if the node is already present
                merged_nodes[node_key].metadata["weighted_score"] += node.metadata["weighted_score"]
            else:
                merged_nodes[node_key] = node
        
        final_nodes = list(merged_nodes.values())
        final_nodes.sort(key=lambda node: node.metadata["weighted_score"], reverse=True)

        ##################################################################
        ##### Limit the number of hits to the specified token limit  #####
        ##################################################################
        
        max_token_limit = TOKENS_PER_DOCUMENT * len(target_files)
        
        max_token_results = []
        current_token_sum = 0
        
        for one_node in final_nodes:
            tokens_needed = num_tokens_from_string(one_node.text, "gpt-4o-mini")
            if current_token_sum + tokens_needed <= max_token_limit:
                max_token_results.append(one_node)
                current_token_sum += tokens_needed
            else:
                # If adding this item exceeds the max token count -> break
                # print("WE HIT MAXIMUM TOKEN LIMIT")
                break
        
        final_results = max_token_results[:number_of_hits]
        
        ##################################################################
        ########## Remove hits below min_hit_score, if needed  ###########
        ##################################################################
        
        if min_hit_score is not None: 
            final_results_pruned = []
            for one_result in final_results:
                if one_result.metadata["weighted_score"] >= min_hit_score:
                    final_results_pruned.append(one_result)
            
            final_results = final_results_pruned
        
        return final_results


    def initialize_BM25(self, nodes: list[TextNode]) -> VectorStoreIndex:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=80,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )
        return bm25_retriever



if __name__ == "__main__":
    search_obj = DocSearch('./data/AzulSa/txt')
    search_results = search_obj.hybrid_search("inconsistency", ["AzulSa_20170303_F-1A_EX-10.3_9943903_EX-10.3_Maintenance Agreement2.txt"], 5)
    
    for res in search_results:
        print("res: ", search_results)

# bm25_retriever.persist("./bm25_retriever")
# loaded_bm25_retriever = BM25Retriever.from_persist_dir("./bm25_retriever")



