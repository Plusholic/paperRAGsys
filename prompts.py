"""Default prompt for ReAct agent."""


# ReAct chat prompt
# TODO: have formatting instructions be a part of react output parser

REACT_CHAT_SYSTEM_HEADER = """\

You are designed to help with a variety of tasks, from answering questions \
    to providing summaries to other types of analyses.

You are assistant to a researcher. You are asked to find papers related to a topic.
You must use the tool to answer.

## Tools
Tool to find context and answer based on that context.
Based on the given context, if you don't know, tell me you don't know.
previous knowledges are not allowed. just answer 'I don't know' if you don't know."

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""

DEFAULT_TEXT_QA_PROMPT_TMPL = """
{context_str}
Query : {query_str}
Answer : 
"""


# DEFAULT_TEXT_QA_PROMPT_TMPL = """

# You are assistant to a researcher.
# The context below is part of a paper.
# In the context below, find the information that is relevant to your query.
# Don't use prior knowledge, and answer based on the context given.

# {context_str}
# Query: {query_str}\n"
# Answer: 
# """

# You need to find and present all the Contexts that correspond to your query.
# Slowly reconsider whether the information is correct
# Do the keywords in the query exist in context?
# The following paper contains the content of the query ref by [Title]
# Find information related to the query in the context below and answer based on it.
# if you don't know, answer shortly just 'I don't know' instead of another description
# When you respond, Based on the given context, you must indicate which text in the referenced context you are basing your answer on.
# ```
# Context : [If you can't find the answer to a query in context]
# Answer : The given context has no information relevant to the query.
# ```

# DEFAULT_REFINE_PROMPT_TMPL = (
# """
# ## Note
# The original query is as follows: {query_str}
# We have provided an existing answer: {existing_answer}
# We have the opportunity to refine the existing answer
# Answer with the title of the context you referenced, separated by a number. In context, the title is denoted by ##.
# (only if needed) with some more context below.

# ## Context : 
# ```
# {context_msg}
# ```

# ## Output Format
# ```
# Context : [If you can find the answer to a query in context]
# Refined Answer : The following paper contains the content of the query 1. [Title], 2. [Title], ...
# ```

# ```
# Context : [If you can't find the answer to a query in context]
# Refined Answer : The given context has no information relevant to the query.
# ```

# Query: {query_str}\n"
# Refined Answer: 
# """    

# )

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "You need to refine your existing answer."
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the query. "
    "When answering a query, you must include a Title to base your answer on. Titles are separated by ## in context."
    "Synthesizes the given contextual information to answer the query."
    
    "Refined Answer: "
)

    # "If the context isn't useful, return the original answer.\n"
    # "When answering a query, you must include the title information, separated by ##."


DEFAULT_TREE_SUMMARIZE_TMPL = (
    "The context below is the title of the article, and the answer to the query referencing that article."
    "Given the information from multiple sources and not prior knowledge, answer the query.\n"
    "When answering, be sure to include a ## separated title in your answer."
    
    'Example format: \n'
    "----------------------\n"
    "## 0. Title : <Title 0>\n"
    "<Answer about query of <Title 0>\n"
    "## 1. Title : <Title 1>\n"
    "<Answer about query of <Title 1>\n"
    "...\n\n"
    "## n. Title : <Title n>\n"
    "<Answer about query of <Title n>\n"
    "----------------------\n"
    "Query : <query>\n"
    "Answer : \n"
    "<Title 1>, <Title 3>... <Title n> are the papers that contain the content of the query.\n"
    "Given that, you might want to consider using the <Your Answer about query is Here>"
    "\n\n"
    "Please answer in the format above.\n"
    "Find all titles that can answer the query."
    "Let's try this now: \n\n"
    "----------------------\n"
    "{context_str}\n"
    "----------------------\n"
    "Query : {query_str}\n"
    "Answer : \n"
)


QUERY_GEN_PROMPT = (
"You are a thesis search assistant."
"A user has a question about the content of your paper.A user has a question about the content of your paper."
"Create {num_queries} search queries, one on each line, to help users find the information they need."
"Must be related to the input query."
"Query : {query}\n"
"Queries : \n"
)