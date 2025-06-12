import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy
import numpy as np
import faiss
from tqdm import tqdm
import ujson as json


class NumpySearcher(object):
    def __init__(self, reference_vectors):
        self.reference_vectors = reference_vectors

    def search(self, query_vector, final_num_neighbors=5):
        scores = []
        for tmp_reference in self.reference_vectors:
            scores.append(np.dot(query_vector, tmp_reference))
        scores = np.asarray(scores)
        sort_index = numpy.argsort(scores)
        sort_index = sort_index[::-1]
        return_values = list()
        for i in sort_index[:final_num_neighbors]:
            return_values.append(scores[i])
        return sort_index[:final_num_neighbors], return_values

    def search_batch(self, query_vectors, final_num_neighbors=5):
        indexes = list()
        values = list()
        for tmp_query in query_vectors:
            tmp_indexes, tmp_values = self.search(tmp_query, final_num_neighbors)
            indexes.append(tmp_indexes)
            values.append(tmp_values)
        return indexes, values


class KR(object):
    """
    The Knowledge Retrieval Module
    """

    def __init__(self, sentences=None, embeddings=None, searcher_name="FAISS"):
        if len(embeddings) < 16:
            searcher_name = "Numpy"
        
        if searcher_name == "SCANN":
            # SCANN does not support M1 chips
            self.searcher = scann.scann_ops_pybind.builder(embeddings, 10, "dot_product").tree(
                num_leaves=max(1, min(2000, len(embeddings) - 1)), num_leaves_to_search=100,
                training_sample_size=len(embeddings)).score_ah(
                2, anisotropic_quantization_threshold=0.2).reorder(
                max(1, min(2000, len(embeddings) - 1))).build()
            raise NotImplementedError
        elif searcher_name == "FAISS":
            self.searcher = faiss.IndexFlatL2(embeddings.shape[1])
            self.searcher.add(embeddings)
        elif searcher_name == "Numpy":
            self.searcher = NumpySearcher(embeddings)
        else:
            raise NotImplementedError

        self.searcher_name = searcher_name
        self.sentences = sentences


    # deleted on 11/13/2023:
    # knowledge_retrieval, sentence_encoding, knowledge_retrieval_batch, knowledge_retrieval_embedding_batch,
    # sentence_encoding_batch

    def knowledge_retrieval_embedding(self, input_embedding, num_of_knowledge=5):
        """
        :param input_embedding: Input embedding to be used as the retrieval key
        :type input_embedding: numpy array
        :param batch_size: Batch size to be used by the retrieval
        :type batch_size: int
        :param num_of_knowledge: Number of returned knowledge
        :type num_of_knowledge: int
        :param progress_bar_header: Display header used for the tqdm bar
        :type progress_bar_header: str
        :return: list of retrieved knowledge
        """
        if self.searcher_name == "SCANN":
            neighbors, similarities = self.searcher.search(input_embedding, final_num_neighbors=num_of_knowledge)
        elif self.searcher_name == "FAISS":
            similarities, neighbors = self.searcher.search(numpy.asarray([input_embedding]), num_of_knowledge)
            neighbors = neighbors[0]
            similarities = similarities[0]
        elif self.searcher_name == "Numpy":
            neighbors, similarities = self.searcher.search(input_embedding, final_num_neighbors=num_of_knowledge)
        else:
            raise NotImplementedError

        extracted_result = []
        for i, tmp_neighbor in enumerate(neighbors):
            extracted_result.append({"sentence_key": self.sentences[tmp_neighbor], "similarity": str(similarities[i])})
        return extracted_result
