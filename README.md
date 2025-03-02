# Arash Dabiri's solution to the problem

I show how the code works in "test_solution.ipynb", but here is an overview as well:


## Overview of solution

My tech stack is LlamaIndex, BM25, OpenAI (gpt-4o-mini throughout).   

I put most focus on the search engine, as the LLM only can answer as well as the data it is given. Therefore hybrid search has been implemented to both use contextual info (embeddings) and also search on keywords (BM25). The User question is also rewritten before search, to make sure that the wording doesn't negatively affect search.    

search_engines.py contains all search related code, backend.py contains the other two classes needed to run the code.    

#### The steps are:

1. Chunking  
1.1. Text is chunked by sentences

2. Search engine  
2.1. User question is rewritten (by GPT-4o-mini) to remove superfluous words -> improving search  
2.2 Hybrid search is used, with equal weight on both BM25 (keyword search) and vector search (text embeddings). There is a limitation on number of search hits to not go over token limit.

3. Answering  
3.1. GPT-4o-mini is fed search results and is instructed to reason and then answer. There is no strict limitation on output format, _focus is to reason/think before answering_.  
3.2. A second call to GPT-4o-mini is used to re-format answer to suit the output format.  



Feel free to check out "test_solution.ipynb" to see the code in action!
