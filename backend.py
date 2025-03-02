from openai import OpenAI
from llama_index.core.schema import NodeWithScore
from search_engines import DocSearch


class ColumnDataObj:
    # Define allowed data types and their descriptions
    allowed_data_types = {
        "Text": "Free form text",
        "Date": "Formatted date (ISO-8601), for example: 2025-02-25",
        "Boolean": "Yes/No (A definite Yes or No, and nothing else)",
        "Currency": "Currency (e.g. 500 SEK, 30 USD)"
    }

    def __init__(self, prompt, data_type):
        # Check that data_type is valid
        if data_type not in self.allowed_data_types:
            raise ValueError(
                f"data_type must be one of {list(self.allowed_data_types.keys())}"
            )
        self.prompt = prompt
        self._data_type = data_type


    def get_data_type(self):
        return self._data_type
    
    def get_data_type_prompt(self):
        # Return the description associated with the data type
        return self.allowed_data_types[self._data_type]

    def get_prompt(self):
        return self.prompt

    def get_allowed_data_types(self):
        return self.allowed_data_types
    


class AnswererObj:    
    def __init__(self) -> None:
        self.client = OpenAI()
        
    def prompt_to_search_query(self, original_prompt: str) -> str:
        
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are search query writer. You have access to a database with hundreds of text files, and write short, informative search queries based on user prompts."},
                {"role": "user", "content": "What are the restrictions imposed on trade/sales?"},
                {"role": "assistant", "content": "trade sales restrictions"},
                {"role": "user", "content": "What is the date when the agreement went into force?"},
                {"role": "assistant", "content": "date agreement"},
                {"role": "user", "content": original_prompt},
            ]
        )
        return completion.choices[0].message.content
    
    
    def search_results_to_string(self, search_results: list[NodeWithScore]) -> str:
        all_results_string = "Search results:"
        for index, one_search_res in enumerate(search_results):
            all_results_string = all_results_string + "\n" + f"----- Result: {index} -----"
            all_results_string = all_results_string + "\n" + one_search_res.text
        
        return all_results_string
    
    
    def format_gpt_response(self, input_text: str, columnObj: ColumnDataObj) -> str:

        
        output_format = columnObj.get_data_type()

        if output_format == "Text": # If in text format, we only want to reduce the "wordiness" of the answer
            instruction_prompt = f"""
            QUESTION: What is the governing law of the contract?
            UNFORMATTED ANSWER:
            To determine the governing law of the contract, we need to look for clauses related to the jurisdiction or choice of law specified in the contract. These clauses are often found under headings such as "Governing Law," "Jurisdiction," or "Applicable Law."

            Reviewing the search results provided:

            1. **Result 0** and **Result 3** mainly discuss miscellaneous provisions of the contract, such as waiver, amendment procedures, and confidentiality, and do not mention governing law.
            
            2. **Result 1** contains key information under section "24. GOVERNING LAW AND ARBITRATION." It states:
            - **Governing Law:** The text specifies that, in accordance with Section 5-1401 of the New York General Obligations Law, the agreement and any potential claims arising therefrom "shall be governed by, and construed in accordance with, the laws of the State of New York, U.S.A." It also explicitly notes that this governance applies exclusively, excluding Section 7-101 of the New York General Obligations Law from applicability.
            
            3. **Result 2** provides a brief context of the agreement's parties and overview but no specific reference to governing law.

            4. **Result 4** doesn't mention governing law directly either but notes the binding nature of the agreement and possible assignment considerations.

            Based on this analysis, **Result 1** clearly outlines the governing law, which is the laws of the State of New York, U.S.A. This response is derived explicitly from section 24.1 of the result.

            Thus, the governing law of the contract is the laws of the State of New York, U.S.A., as stipulated by the contract.
            
            FORMATTED ANSWER:
            """
            
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an answer rewriter, rewriting answers to legal questions to make the answers (marked as UNFORMATTED ANSWER) less wordy - and focus on answering the question (marked as QUESTION). Only write the FORMATTED ANSWER."},
                        {"role": "user", "content": instruction_prompt},
                        {"role": "assistant", "content": "The governing law of the contract is the laws of the State of New York, U.S.A., as stipulated by the contract."},
                        {"role": "user", "content": f"QUESTION: {columnObj.get_prompt()}\n UNFORMATTED ANSWER: {input_text}\n FORMATTED ANSWER:\n"},
                    ],
                temperature=0  
            )
            return completion.choices[0].message.content
        
        else:
            # Make the API call
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an answer rewriter, rewriting answers to legal questions to suit a certain hard-set format. You only write out answers in the correct answer format, nothing else."},
                        {"role": "user", "content": f"Answer: No, there is not a limitation of liability clause. \n\nOutput format: Boolean - {columnObj.get_allowed_data_types()['Boolean']}"},
                        {"role": "assistant", "content": "No"},
                        {"role": "user", "content": f"Answer: The limitations mention January 1st of 2024.  \n\nOutput format: Date - {columnObj.get_allowed_data_types()['Date']}"},
                        {"role": "assistant", "content": "2024-01-01"},
                        {"role": "user", "content": f"Answer: In paragraph 3.1 5000 USD is mentioned.  \n\nOutput format: Currency - {columnObj.get_allowed_data_types()['Currency']}"},
                        {"role": "assistant", "content": "5000 USD"},
                        {"role": "user", "content": f"Answer: {input_text}  \n\nOutput format: {output_format} - {columnObj.get_allowed_data_types()[output_format]}"},
                    ],
                temperature=0  
            )

            return completion.choices[0].message.content
        

    def answer(self, search_obj: DocSearch, search_file_list: list[str], columnObj: ColumnDataObj) -> str:
        
        # Rewrite prompt to search query
        search_query = self.prompt_to_search_query(columnObj.get_prompt())
        print("search_query: ", search_query)
        
        # Search for results given prompt
        search_results = search_obj.hybrid_search(search_query, search_file_list, 15)
        
        print("Number of search hits:", len(search_results))
        # for one_res in search_results:
        #     print("TEXT of one search hits:\n", one_res.text)
            
        # for one_res in search_results:
        #     print("Scores of one search hits: ", one_res.metadata["weighted_score"])
        # print("----------------")
        
        search_results_as_readable_string = self.search_results_to_string(search_results) # Make it readable
        
        # Give results + output format to GPT
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You read through legal texts and answer questions regarding these. It is of outmost importance that your answer is based on the texts provided."},
                {"role": "user","content": f"Question: {columnObj.get_prompt()}"},
                {"role": "user","content": f"Sources: {search_results_as_readable_string}"},
                {"role": "user","content": f"Read through the sources above. Think first aloud - perhaps by listing your thoughts into bullets. Then, at last, answer the question based on your previous reasoning."},
            ]
        )
        
        unformatted_answer = completion.choices[0].message.content
        
        print("Unformatted answer: ", unformatted_answer)
        print("----------------")
        
        formatted_answer = self.format_gpt_response(unformatted_answer, columnObj)
        
        return formatted_answer
    
    
