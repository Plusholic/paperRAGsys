


#######################################################
######################## Utils ########################
#######################################################

"""Utils for pretty print."""
import textwrap
from llama_index.response.schema import Response
from llama_index.schema import NodeWithScore
from llama_index.utils import truncate_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize    

# def pprint_source_node(
#     source_node: NodeWithScore, source_length: int = 350, wrap_width: int = 70
# ) -> None:
#     """Display source node for jupyter notebook."""
#     source_text_fmt = truncate_text(
#         source_node.node.get_content().strip(), source_length
#     )
#     print(f"Node ID: {source_node.node.node_id}")
#     try:
#         print(f"File Name: {source_node.node.metadata['file_name']}")
#     except:
#         pass
#     print(f"Similarity: {source_node.score}")
#     print(textwrap.fill(f"Text: {source_text_fmt}\n", width=wrap_width))

def pprint_response_title(
    response: Response,
    query = None,
    source_length: int = 350,
    wrap_width: int = 70,
    show_source: bool = False,
) -> None:
    """Pretty print response for jupyter notebook."""
    if response.response is None:
        response_text = "None"
    else:
        response_text = response.response.strip()

    response_text = f"Final Response: {response_text}"
    # 한 문든을 모든 줄의 길이가 최대 width 자가 되도록 래핑
    print(textwrap.fill(response_text, width=wrap_width))

    if show_source:
        for ind, source_node in enumerate(response.source_nodes):
            try:
                print("_" * wrap_width)
                print(f"Source Node : {ind + 1}/{len(response.source_nodes)}")
                print(textwrap.fill(f"File Name : {source_node.metadata['file_name']}", width=wrap_width))
                # pprint_source_node(
                #     source_node, source_length=source_length, wrap_width=wrap_width
                # )
                # query = None 이면 source node가 그냥 프린트됨
                pprint_res = sim_sentence_extract(query, source_node, source_length)
                print(textwrap.fill(f"Text : {pprint_res}", width=wrap_width))
            except:
                ind -= 1
                pass
            
def sim_sentence_extract(query, source_node, source_length):

    if query == None:
        return source_node
    else:
        paragraph = source_node.get_content()
    
    # 문단을 문장 단위로 분할
    sentences = sent_tokenize(paragraph)

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # 쿼리를 TF-IDF 벡터화
    query_vector = vectorizer.transform([query])

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # 가장 유사한 문장을 찾기
    most_similar_sentence_index = cosine_similarities.argmax()
    if most_similar_sentence_index > 1:
        return ' '.join(sentences[most_similar_sentence_index-2:most_similar_sentence_index+2])[:source_length]
    else:
        return ' '.join(sentences[:most_similar_sentence_index+3])[:source_length]



#######################################################
#################### Postprocessing ###################
#######################################################

from llama_index import QueryBundle
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore
from llama_index.schema import TextNode
from collections import defaultdict
from typing import Callable, List, Optional 
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.utils import globals_helper
from llama_index.schema import MetadataMode

class UnionNodePostprocessor(BaseNodePostprocessor):
    # 같은 파일에서 검색된 노드를 합치는 후처리기
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        node_dict = defaultdict(str)
        score_dict = defaultdict(int)
        cnt_dict = defaultdict(int)
        
        
        for n in nodes: # score filtering need
            node_dict[n.metadata['file_name']] += "\n---------------------------\n"
            node_dict[n.metadata['file_name']] += "---------------------------\n\n"
            node_dict[n.metadata['file_name']] += n.get_content()

            score_dict[n.metadata['file_name']] += n.score
            cnt_dict[n.metadata['file_name']] += 1

        node_list = []
        # score update
        
        for file_name in node_dict:
            score_dict[file_name] /= cnt_dict[file_name]
            node_list.append(NodeWithScore(node=TextNode(text=node_dict[file_name],
                                                         metadata={'file_name' : file_name}),
                                           score=score_dict[file_name]))
        
        return node_list


class UnionNodePostprocessorSortedScore(BaseNodePostprocessor):
    # 같은 파일에서 검색된 노드를 합치는 후처리기
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        # node_dict = defaultdict(str)
        score_dict = defaultdict(int)
        # cnt_dict = defaultdict(int)
        tmp_data = list((nodes[i].metadata['file_name'],
                         nodes[i].text,
                         nodes[i].score
                         ) for i in range(nodes.__len__()))
        
        grouped_filename_sorted_dict = defaultdict(str)
        grouped_filename_dict = defaultdict(list)
        for file_name, contents, score in tmp_data:
            grouped_filename_dict[file_name].append((contents, score))
        
        for file_name in grouped_filename_dict:
            score = 0
            sorted_data = sorted(grouped_filename_dict[file_name], key=lambda x:x[1], reverse=True) # x[2] : score
            
            # grouped_filename_sorted_dict[file_name] += f"\n ## Title : {file_name} ##\n\n"
            
            for i in range(sorted_data[:4].__len__()):
                # grouped_filename_sorted_dict[file_name] += f"\n==========================="
                
                # grouped_filename_sorted_dict[file_name] += '\n'
                grouped_filename_sorted_dict[file_name] += sorted_data[:4][i][0]
                score_dict[file_name] += score
            score_dict[file_name] /= (i+1)
        

        node_list = []
        # score update
        
        for file_name in grouped_filename_sorted_dict:
            node_list.append(NodeWithScore(node=TextNode(text=grouped_filename_sorted_dict[file_name],
                                                         metadata={'file_name' : file_name}),
                                           score=score_dict[file_name]))
        
        return node_list




class DuplecatedNodePostprocessor(BaseNodePostprocessor):
    # 검색된 노드의 중복 컨텐츠를 제거하는 후처리기
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:

        contents_list = list()
        metadata_list = list()
        scores_list = list()
        for n in nodes:
            
            if n.get_content() not in contents_list: # metadata는 같은 파일의 다른 부분일 수 있음.

                contents_list.append(n.get_content())
                try:
                    metadata_list.append(n.metadata['file_name'])
                except:
                    metadata_list.append(n.node.source_node.metadata['file_name'])

                scores_list.append(n.score)
        
        node_list = []
        for content, metadata, score in zip(contents_list, metadata_list, scores_list):
            node_list.append(NodeWithScore(node=TextNode(text = content,
                                                         metadata = {'file_name' : metadata}),
                                           score = score)
                             )
        
        return node_list

class LimitRetrievedNodesLength:

    def __init__(self, limit: int = 3000, tokenizer: Optional[Callable] = None):
        self._tokenizer = tokenizer or globals_helper.tokenizer
        self.limit = limit

    def postprocess_nodes(self, nodes, query_bundle):
        included_nodes = []
        current_length = 0

        for node in nodes:
            current_length += len(self._tokenizer(node.node.get_content(metadata_mode=MetadataMode.LLM)))
            if current_length > self.limit:
                break
            included_nodes.append(node)

        return included_nodes
    
#######################################################
#################### Preprocessing ####################
#######################################################
 
from llama_index import Document
import re
# |Reference|REFER ENCES|RefeRences|reFerences|Referen ces|r  e  f  e  r  e  n  c  e  s|Refer ences|EFERENCES
def remove_reference_part(pattern, documents, filter_str = '1'):

    except_list = []
    else_list = []
    processing_doc_list = []

    for i in range(documents.__len__()):
        
        matches = re.search(pattern, documents[i].get_content().replace('\n',' '), flags=re.IGNORECASE) # \n이 있으면 정규표현식에서 안 찾아짐
        try:
            # print(matches.groups()[1], matches.groups()[2])
            pass
            
            for ref_num in matches.groups()[2][:10]:
                # 레퍼런스가 숫자로 시작하는 경우
                if ref_num == filter_str:
                    processing_doc_list.append(
                        Document(
                            id_ = documents[i].id_,
                            text=matches.groups()[0],
                            metadata={'file_name' : documents[i].metadata['file_name']}
                            )
                    )
                    break
            else:
                # 숫자로 시작하지 않는 경우, 여기에 정말 reference 부분이 없다면, else_list는 다시 전처리 해야 함.
                # print(ref_num)
                else_list.append(documents[i])
            
        except:
            # 그룹으로 detect 못 한 부분, Documents에 text=None은 에러나기 때문
            except_list.append(documents[i])

    print("result")
    print("processing cnt :\t",processing_doc_list.__len__())
    print("filter_str not recognize cnt :\t", else_list.__len__())
    print("pattern not recognize cnt :\t", except_list.__len__())
    return processing_doc_list, except_list, else_list



#######################################################


from typing import List, Optional, Sequence, cast

from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.output_parsers.base import StructuredOutput
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.prompts.mixin import PromptDictType
from llama_index.prompts.prompt_type import PromptType
from llama_index.question_gen.output_parser import SubQuestionOutputParser
from llama_index.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    build_tools_text,
)
from llama_index.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.schema import QueryBundle
from llama_index.service_context import ServiceContext
from llama_index.tools.types import ToolMetadata
from llama_index.types import BaseOutputParser


class CustomQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        llm_predictor: BaseLLMPredictor,
        prompt: BasePromptTemplate,
    ) -> None:
        self._llm_predictor = llm_predictor
        self._prompt = prompt

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "CustomQuestionGenerator":
        # optionally initialize defaults
        service_context = service_context or ServiceContext.from_defaults()
        prompt_template_str = prompt_template_str or DEFAULT_SUB_QUESTION_PROMPT_TMPL
        output_parser = output_parser or SubQuestionOutputParser()

        # construct prompt
        prompt = PromptTemplate(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SUB_QUESTION,
        )
        return cls(service_context.llm_predictor, prompt)

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"question_gen_prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "question_gen_prompt" in prompts:
            self._prompt = prompts["question_gen_prompt"]

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm_predictor.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )
        # print(query_str)
        # print(tools_str)
        # print(tools)
        # print('여기까지 왔다')
        # print(prediction)
        
        prediction = [
            
            {
                "sub_question": query_str,
                "tool_name" : tool.name 
            } for tool in tools
        ]
        import json
        prediction = json.dumps(prediction)
        # print(prediction)
        
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        # prediction = await self._llm_predictor.apredict(
        #     prompt=self._prompt,
        #     tools_str=tools_str,
        #     query_str=query_str,
        # )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(query_str)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output


