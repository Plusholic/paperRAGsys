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
            
            grouped_filename_sorted_dict[file_name] += f"\n ## Title : {file_name} ##\n\n"
            
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