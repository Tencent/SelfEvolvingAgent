import ast
import concurrent
import copy
import itertools
import json
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import requests
from tqdm import tqdm
from cognitive_kernel.memory_kernel.knowledge_engine_connection import (
    DOCUMENT_TABLE_NAME,
    DOCUMENT_COLUMNS,
    DOCUMENT_COLUMN_TYPES,
    PROPOSITION_TABLE_NAME,
    PROPOSITION_COLUMNS,
    PROPOSITION_COLUMN_TYPES,
    DOCUMENT_PROPOSITION_MAPPING_TABLE_NAME,
    DOCUMENT_PROPOSITION_MAPPING_COLUMNS,
    DOCUMENT_PROPOSITION_MAPPING_COLUMN_TYPES,
    PROPOSITION_CONCEPT_MAPPING_TABLE_NAME,
    PROPOSITION_CONCEPT_MAPPING_COLUMNS,
    PROPOSITION_CONCEPT_MAPPING_COLUMN_TYPES,
    PROPOSITION_RELATION_TABLE_NAME,
    PROPOSITION_RELATION_COLUMNS,
    PROPOSITION_RELATION_COLUMN_TYPES,
    CONCEPT_TABLE_NAME,
    CONCEPT_COLUMNS,
    CONCEPT_COLUMN_TYPES,
    CONCEPT_ABSTRACTION_MAPPING_TABLE_NAME,
    CONCEPT_ABSTRACTION_MAPPING_COLUMNS,
    CONCEPT_ABSTRACTION_MAPPING_COLUMN_TYPES,
    MENTION_CONCEPT_MAPPING_TABLE_NAME,
    MENTION_CONCEPT_MAPPING_COLUMNS,
    MENTION_CONCEPT_MAPPING_COLUMN_TYPES,
    KnowledgeEngineConnection,
)
from cognitive_kernel.memory_kernel.retriever import KR
from collections import defaultdict

MAX_PROCESS = 1024


def generate_module_id():
    return uuid.uuid4().hex[:16]


def chunk_doc(doc, chunk_size=300):
    """
    This function will chunk doc into fixed size.
    :param doc: input document
    :type doc: str
    :param chunk_size: the max_size of each chunk
    :type chunk_size: int
    :return:
    """
    if len(doc) < chunk_size:
        return [doc]
    all_sentences = doc.split("。")
    sentences_by_batch = list()
    tmp_query = ""
    for tmp_s in all_sentences:
        if len(tmp_query) + len(tmp_s) < chunk_size:
            tmp_query += tmp_s + "。"
        else:
            sentences_by_batch.append(tmp_query)
            tmp_query = tmp_s + "。"
    if len(tmp_query) > 0:
        sentences_by_batch.append(tmp_query)

    return sentences_by_batch


class KnowledgeEngineConfig(object):
    def __init__(self, config_dict=None):
        self.db_name = None
        self.db_type = None
        self.db_path = None
        self.chunk_size = None
        self.neighbor_size = None

        # added on 3/7/2024
        self.retrieval_type = "doc"

        if config_dict:
            self.update_config(config_dict)

    def update_config(self, config_dict):
        """
        This function updates the config file with a dictionary
        :param config_dict: config file in the form of dictionary
        :return: None
        """
        if "db_name" in config_dict:
            self.db_name = config_dict["db_name"]
        if "db_type" in config_dict:
            self.db_type = config_dict["db_type"]
        if "db_path" in config_dict:
            self.db_path = config_dict["db_path"]
        if "chunk_size" in config_dict:
            self.chunk_size = config_dict["chunk_size"]
        if "neighbor_size" in config_dict:
            self.neighbor_size = config_dict["neighbor_size"]

        # added on 3/7/2024
        if "retrieval_type" in config_dict:
            self.retrieval_type = config_dict["retrieval_type"]


class KnowledgeEngine(object):
    def __init__(self, args, inference_urls=None, mode="inference"):
        """
        This class is the main system of the KnowledgeEngine project. It manages the global knowledge engine modules
        and local ones.
        :param args: arguments we need to start the knowledge engine service
        """
        self.args = args
        self.num_local_db = self.args.num_local_module
        self.local_module_path = self.args.local_module_path
        self.inference_urls = inference_urls
        self.global_id2module = dict()
        self.global_id2config = dict()
        self.local_id2module = dict()
        self.local_id2config = dict()

        # Loading the global_kbs
        with open(self.args.global_config_file_path) as fin:
            for line in fin:
                current_config = KnowledgeEngineConfig(json.loads(line))
                current_id = current_config.db_name
                print("processing the db:", current_id)
                self.global_id2config[current_id] = current_config
                self.global_id2module[current_id] = KnowledgeEngineModule(
                    config=current_config,
                    inference_urls=self.inference_urls,
                    mode=mode,
                    module_type="global",
                )

        # Loading the default config
        with open(self.args.default_config_file_path) as fin:
            self.default_config = KnowledgeEngineConfig(json.load(fin))

    def setup_global_knowledge_engine_module(self, config_file, mode="inference"):
        current_id = config_file.db_name
        print("processing the db:", current_id)
        self.global_id2config[current_id] = config_file
        self.global_id2module[current_id] = KnowledgeEngineModule(
            config=config_file,
            inference_urls=self.inference_urls,
            mode=mode,
            module_type="global",
        )

    def get_local_knowledge_engine_module(self, config_file=None, module_name=None):
        """
        This function will create a local knowledge engine module. If the config_file is a valid config file, we will create
        the knowledge engine module accordingly. Otherwise, we will randomly start an empty local connection.
        :param module_name:
        :param config_file:
        :return:
        """
        if module_name is not None and module_name in self.local_id2module:
            return self.local_id2module[module_name]

        if isinstance(config_file, KnowledgeEngineConfig):
            # we will start a knowledge engine module connection with the provided config_file
            current_id = config_file.db_name
            if current_id not in self.local_id2config:
                self.local_id2config[current_id] = config_file
                self.local_id2module[current_id] = KnowledgeEngineModule(
                    config=config_file,
                    inference_urls=self.inference_urls,
                    mode="inference",
                    module_type="local",
                )
            return self.local_id2module[current_id]
        elif config_file is None:
            # we need to randomly start an empty knowledge engine module

            if module_name is None:
                current_id = generate_module_id()
            else:
                current_id = module_name
            current_db_path = self.local_module_path + current_id + ".db"
            print("We are starting a new tmp_db with path:", current_db_path)
            tmp_config = copy.deepcopy(self.default_config)
            tmp_config.db_name = current_id
            tmp_config.db_path = current_db_path
            self.local_id2config[current_id] = tmp_config
            self.local_id2module[current_id] = KnowledgeEngineModule(
                config=self.local_id2config[current_id],
                inference_urls=self.inference_urls,
                mode="inference",
                module_type="local",
            )
            return self.local_id2module[current_id]
        else:
            raise NotImplementedError("We currently do not support this function")

    def get_local_knowledge_engine_module_fresh(
        self, config_file=None, module_name=None
    ):
        """
        This function will create a local knowledge engine module. If the config_file is a valid config file, we will create
        the knowledge engine module accordingly. Otherwise, we will randomly start an empty local connection.
        :param config_file:
        :return:
        """
        if module_name is not None and module_name in self.local_id2module:
            print("loading db:", module_name)
            return self.local_id2module[module_name]

        if isinstance(config_file, KnowledgeEngineConfig):
            # we will start an knowledge engine module connection with the provided config_file
            current_id = config_file.db_name
            self.local_id2config[current_id] = config_file
            self.local_id2module[current_id] = KnowledgeEngineModule(
                config=config_file,
                inference_urls=self.inference_urls,
                mode="inference",
                module_type="local",
            )
            return self.local_id2module[current_id]
        elif config_file is None:
            # we need to randomly start an empty knowledge engine module

            if module_name is None:
                current_id = generate_module_id()
            else:
                current_id = module_name
            current_db_path = self.local_module_path + current_id + ".db"
            print("We are starting a new tmp_db with path:", current_db_path)
            tmp_config = copy.deepcopy(self.default_config)
            tmp_config.db_name = current_id
            tmp_config.db_path = current_db_path
            self.local_id2config[current_id] = tmp_config
            self.local_id2module[current_id] = KnowledgeEngineModule(
                config=self.local_id2config[current_id],
                inference_urls=self.inference_urls,
                mode="inference",
                module_type="local",
            )
            return self.local_id2module[current_id]
        else:
            raise NotImplementedError("We currently do not support this function")

    # deleted on 11/13/2023
    # find_answer_single_db, update_local_module

    # modified on 11/03/2023
    # modified on 1/30/2024, dealing with scores
    # modified on 2/9/2024, add metadata
    # modified on 2/15/2024, add keywords
    def find_relevant_info_single_db(
        self,
        db_name,
        query,
        retrieval_mode="doc+prop+concept",
        hard_match_degree=2,
        soft_match_top_k=5,
        sim_threshold=0.01,
        neighbor_size=5,
        filter_doc=False,
    ):
        print(f"self.local_id2module: {self.local_id2module}")
        if db_name in self.global_id2module:
            knowledge_engine_module = self.global_id2module[db_name]
        elif db_name in self.local_id2module:
            knowledge_engine_module = self.get_local_knowledge_engine_module(
                module_name=db_name
            )
        else:
            knowledge_engine_module = self.get_local_knowledge_engine_module_fresh(
                module_name=db_name
            )

        doc_contents = []
        doc_metadata = []
        doc_scores = []
        if (
            "hard" in retrieval_mode
            or "prop" in retrieval_mode
            or "concept" in retrieval_mode
        ):
            docs_1, metadata_1, scores_1 = (
                knowledge_engine_module.find_relevant_knowledge_single_proposition(
                    query,
                    retrieval_mode,
                    hard_match_degree,
                    soft_match_top_k,
                    sim_threshold,
                    neighbor_size,
                )
            )
            doc_contents += docs_1
            doc_metadata += metadata_1
            doc_scores += scores_1
        if "doc" in retrieval_mode or "keyword" in retrieval_mode:
            docs_2, metadata_2, scores_2 = (
                knowledge_engine_module.find_relevant_knowledge_single_document(
                    query, retrieval_mode, soft_match_top_k, sim_threshold
                )
            )
            doc_contents += docs_2
            doc_metadata += metadata_2
            doc_scores += scores_2

        if "+" in retrieval_mode:
            doc2score = defaultdict(float)
            doc2metadata = {}

            # modified on 1/30/2024, doc merging strategy
            for doc, metadata, score in zip(doc_contents, doc_metadata, doc_scores):
                # doc2score[doc] += score  # use plus for merging
                doc2score[doc] = max(doc2score[doc], score)  # use max for merging
                doc2metadata[doc] = metadata

            doc_contents = []
            doc_metadata = []
            doc_scores = []
            for doc in doc2score.keys():
                doc_contents.append(doc)
                doc_metadata.append(doc2metadata[doc])
                doc_scores.append(doc2score[doc])

        # added on 1/9/2024, filter documents
        if filter_doc:
            res = knowledge_engine_module.filter_doc_batch(
                [query] * len(doc_contents), doc_contents
            )
            filtered_doc_contents = []
            filtered_doc_metadata = []
            filtered_scores = []
            for i in range(len(doc_contents)):
                if res[i].strip() == "是":
                    filtered_doc_contents.append(doc_contents[i])
                    filtered_doc_metadata.append(doc_metadata[i])
                    filtered_scores.append(doc_scores[i])
            doc_contents = filtered_doc_contents
            doc_metadata = filtered_doc_metadata
            doc_scores = filtered_scores

        # sort the results
        res = [(c, m, s) for c, m, s in zip(doc_contents, doc_metadata, doc_scores)]
        res = sorted(res, key=lambda x: x[2], reverse=True)

        return [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]


class KnowledgeEngineModule(object):
    def __init__(
        self, config, inference_urls=None, mode="inference", module_type="global"
    ):
        # This is the initial function for Knowledge Engine
        self.config = config
        self.inference_urls = inference_urls
        self.mode = mode
        self.module_type = module_type
        if self.config.db_type == "Sqlite":
            self.knowledge_engine_connection = KnowledgeEngineConnection(
                db_path=self.config.db_path,
                db="sqlite",
                chunksize=self.config.chunk_size,
            )
            self.db_connection = self.knowledge_engine_connection.get_conn()
            self.retriever = None

            # added on 11/13/2023
            self.retriever_doc = None
            self.retriever_cp = None

            # added on 2/15/2024
            self.retriever_keywords = None
        elif self.config.db_type == "Mongo":
            raise NotImplementedError
        else:
            raise NotImplementedError
        if mode == "inference" and module_type == "global":
            try:
                self.create_index(table_name="Propositions", column="key_concept")
            except:
                print("index exists")
            try:
                self.create_index(
                    table_name="PropositionConceptMapping", column="concept_id"
                )
            except:
                print("index exists")
            try:
                self.create_index(
                    table_name="DocumentPropositionsMapping", column="prop_id"
                )
            except:
                print("index exists")
            self.setup_retriever()

        if mode == "inference" and module_type == "local":
            self.setup_retriever()

    # deleted on 11/13/2023:
    # find_answer, _find_answer_single_proposition, _ordered_hard_retrieval_single_proposition, _parse_documents,
    # initialize_all_tables, remove_all_tables, _query_decomposition_batch, _entity_linking_batch,
    # _query_based_compression_batch

    # modified on 11/17/2023
    # modified on 1/29/2024, return a list of zero sim scores
    def _hard_retrieval_single_proposition(
        self, query, hard_match_degree, neighbor_size
    ):
        """This function will conduct the hard retrieval based on a single proposition.

        Args:
            query (str): input query
            hard_match_degree (int, optional): the target hard_match_degree. Defaults to 2.
            neighbor_size (int, optional): the target neighbor_size. Defaults to 5.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            List[str]: retrieved documents
        """

        key_concept_and_perspectives = self._concept_perspective_generation_batch(
            [query]
        )
        if hard_match_degree == -1:
            target_pids = list()
        elif hard_match_degree == 0:
            raise NotImplementedError("We currently do not support this function")
        elif hard_match_degree == 1:
            raise NotImplementedError("We currently do not support this function")
        elif hard_match_degree == 2:
            target_concepts = list()
            for tmp_key_concept_and_perspective in key_concept_and_perspectives:
                # target_concepts.append(tmp_key_concept_and_perspective['concept'])

                if tmp_key_concept_and_perspective is None:
                    target_concepts.append("")
                else:
                    target_concepts.append(tmp_key_concept_and_perspective["concept"])

            target_concepts = list(set(target_concepts))
            target_pids_raw_data = (
                self.db_connection.get_rows_by_single_key_multiple_values(
                    table_name=PROPOSITION_TABLE_NAME,
                    by="key_concept",
                    keys=target_concepts,
                    columns=["_id"],
                )
            )
            target_pids = list(map(lambda x: x["_id"], target_pids_raw_data))
            target_pids = list(set(target_pids))
        elif hard_match_degree == 3:
            target_concepts = list()
            for tmp_key_concept_and_perspective in key_concept_and_perspectives:
                target_concepts.append(tmp_key_concept_and_perspective["concept"])
            target_concepts = list(set(target_concepts))
            target_concept_ids = self.knowledge_engine_connection.get_concept_ids(
                concepts=target_concepts
            )
            target_pids_raw_data = (
                self.db_connection.get_rows_by_single_key_multiple_values(
                    table_name=PROPOSITION_CONCEPT_MAPPING_TABLE_NAME,
                    by="concept_id",
                    keys=target_concept_ids,
                    columns=["prop_id"],
                )
            )
            target_pids = list(map(lambda x: x["prop_id"], target_pids_raw_data))
            target_pids = list(set(target_pids))
        else:
            raise NotImplementedError(
                "We currently do not support this hard_match_degree"
            )
        return target_pids, [0] * len(target_pids)

    # modified on 11/03/2024
    # modified on 1/29/2024, return the scores of retrieved propositions
    def _prop_retrieval_single_proposition(
        self, query, soft_match_top_k, sim_threshold, neighbor_size
    ):
        embeddings = self._encoding_batch_parallel(
            queries=[query], urls=self.inference_urls["sentence_encoding"]
        )
        target_embedding = np.array(json.loads(embeddings[0]), dtype=np.float16)
        if soft_match_top_k > 0:
            raw_results = self.retriever.knowledge_retrieval_embedding(
                input_embedding=target_embedding, num_of_knowledge=soft_match_top_k
            )
            selected_pids = []
            selected_scores = []
            for r in raw_results:
                if float(r["similarity"]) > sim_threshold:
                    selected_pids.append(r["sentence_key"])
                    selected_scores.append(float(r["similarity"]))
            return selected_pids, selected_scores
        else:
            return [], []

    # added on 11/13/2023
    # modified on 1/29/2024, return the scores of retrieved propositions
    def _concept_retrieval_single_proposition(
        self, query, soft_match_top_k, sim_threshold, neighbor_size
    ):
        parse_query = False
        if parse_query:
            cp = self._concept_perspective_generation_batch([query])[0]
            if (
                cp is None
                or not isinstance(cp["concept"], str)
                or not isinstance(cp["perspective"], str)
            ):
                return [], []
            else:
                embeddings = self._encoding_batch_parallel(
                    queries=[cp["concept"] + " " + cp["perspective"]],
                    urls=self.inference_urls["sentence_encoding"],
                )
        else:
            embeddings = self._encoding_batch_parallel(
                queries=[query], urls=self.inference_urls["sentence_encoding"]
            )
        target_embedding = np.array(json.loads(embeddings[0]), dtype=np.float16)
        if soft_match_top_k > 0:
            raw_results = self.retriever_cp.knowledge_retrieval_embedding(
                input_embedding=target_embedding, num_of_knowledge=soft_match_top_k
            )
            selected_pids = []
            selected_scores = []
            for r in raw_results:
                if float(r["similarity"]) > sim_threshold:
                    selected_pids.append(r["sentence_key"])
                    selected_scores.append(float(r["similarity"]))
            return selected_pids, selected_scores
        else:
            return [], []

    # added on 10/24/2023 for document dense retrieval
    # modified on 1/29/2024, return the scores of retrieved docs
    def _doc_retrieval_single_document(self, query, soft_match_top_k, sim_threshold):
        embeddings = self._encoding_batch_parallel(
            queries=[query], urls=self.inference_urls["sentence_encoding"]
        )
        target_embedding = np.array(json.loads(embeddings[0]), dtype=np.float16)
        raw_results = self.retriever_doc.knowledge_retrieval_embedding(
            input_embedding=target_embedding, num_of_knowledge=soft_match_top_k
        )
        selected_dids = []
        selected_scores = []
        for r in raw_results:
            if float(r["similarity"]) > sim_threshold:
                selected_dids.append(r["sentence_key"])
                selected_scores.append(float(r["similarity"]))
        return selected_dids, selected_scores

    # added on 2/15/2024 for keyword dense retrieval
    def _keyword_retrieval_single_document(
        self, query, soft_match_top_k, sim_threshold
    ):
        embeddings = self._encoding_batch_parallel(
            queries=[query], urls=self.inference_urls["sentence_encoding"]
        )
        target_embedding = np.array(json.loads(embeddings[0]), dtype=np.float16)
        raw_results = self.retriever_keywords.knowledge_retrieval_embedding(
            input_embedding=target_embedding, num_of_knowledge=soft_match_top_k
        )
        selected_dids = []
        selected_scores = []
        for r in raw_results:
            if float(r["similarity"]) > sim_threshold:
                selected_dids.append(r["sentence_key"])
                selected_scores.append(float(r["similarity"]))
        return selected_dids, selected_scores

    # added on 10/24/2023 for document dense retrieval
    # modified on 1/29/2024, return the scores of retrieved docs
    # modified on 2/9/2024, add metadata
    # archived on 2/15/2024
    """
    def find_relevant_knowledge_single_document(self, query, soft_match_top_k, sim_threshold):
        retrieved_dids, scores = self._soft_retrieval_single_document(query, soft_match_top_k, sim_threshold)
        document_contents, doc_metadata = self.knowledge_engine_connection.retrieve_doc_content_based_on_doc_ids(retrieved_dids)
        # print('retrieved docs:', document_contents)
        return document_contents, doc_metadata, scores
    """

    # modified on 2/15/2024, add keywords
    def find_relevant_knowledge_single_document(
        self, query, retrieval_mode, soft_match_top_k, sim_threshold
    ):
        retrieved_dids = []
        scores = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if "doc" in retrieval_mode:
                futures_doc_result = executor.submit(
                    self._doc_retrieval_single_document,
                    query,
                    soft_match_top_k,
                    sim_threshold,
                )
                doc_retrieved_pids, doc_scores = futures_doc_result.result()
                retrieved_dids += doc_retrieved_pids
                scores += doc_scores
            if "keyword" in retrieval_mode:
                futures_keyword_result = executor.submit(
                    self._keyword_retrieval_single_document,
                    query,
                    soft_match_top_k,
                    sim_threshold,
                )
                keyword_retrieved_pids, keyword_scores = futures_keyword_result.result()
                retrieved_dids += keyword_retrieved_pids
                scores += keyword_scores

        # merge results
        if "doc" in retrieval_mode and "keyword" in retrieval_mode:
            did2score = defaultdict(float)

            # prop merging strategy
            for did, s in zip(retrieved_dids, scores):
                # did2score[did] += s  # use plus for merging
                did2score[did] = max(did2score[did], s)  # use max for merging

            did_score_list = did2score.items()
            retrieved_dids = [i[0] for i in did_score_list]
            scores = [i[1] for i in did_score_list]

        document_contents, doc_metadata = (
            self.knowledge_engine_connection.retrieve_doc_content_based_on_doc_ids(
                retrieved_dids
            )
        )
        return document_contents, doc_metadata, scores

    # modified on 10/24/2023
    # modified on 1/29/2024, return the scores of retrieved propositions
    # modified on 2/9/2024, add metadata
    def find_relevant_knowledge_single_proposition(
        self,
        query,
        retrieval_mode,
        hard_match_degree,
        soft_match_top_k,
        sim_threshold,
        neighbor_size,
    ):
        retrieved_pids = []
        scores = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if "hard" in retrieval_mode:
                futures_hard_result = executor.submit(
                    self._hard_retrieval_single_proposition,
                    query,
                    hard_match_degree,
                    neighbor_size,
                )
                hard_retrieved_pids, hard_scores = futures_hard_result.result()
                retrieved_pids += hard_retrieved_pids
                scores += hard_scores
            if "prop" in retrieval_mode:
                futures_prop_result = executor.submit(
                    self._prop_retrieval_single_proposition,
                    query,
                    soft_match_top_k,
                    sim_threshold,
                    neighbor_size,
                )
                prop_retrieved_pids, prop_scores = futures_prop_result.result()
                retrieved_pids += prop_retrieved_pids
                scores += prop_scores
            if "concept" in retrieval_mode:
                futures_concept_result = executor.submit(
                    self._concept_retrieval_single_proposition,
                    query,
                    soft_match_top_k,
                    sim_threshold,
                    neighbor_size,
                )
                concept_retrieved_pids, concept_scores = futures_concept_result.result()
                retrieved_pids += concept_retrieved_pids
                scores += concept_scores

        # merge results
        if (
            ("hard" in retrieval_mode and "prop" in retrieval_mode)
            or ("hard" in retrieval_mode and "concept" in retrieval_mode)
            or ("prop" in retrieval_mode and "concept" in retrieval_mode)
        ):
            pid2score = defaultdict(float)

            # modified on 1/30/2024, prop merging strategy
            for pid, s in zip(retrieved_pids, scores):
                # pid2score[pid] += s  # use plus for merging
                pid2score[pid] = max(pid2score[pid], s)  # use max for merging

            pid_score_list = pid2score.items()
            retrieved_pids = [i[0] for i in pid_score_list]
            scores = [i[1] for i in pid_score_list]

        doc_contents, doc_metadata, doc_scores = (
            self.knowledge_engine_connection.retrieve_doc_content_based_on_proposition_ids(
                retrieved_pids, scores
            )
        )
        return doc_contents, doc_metadata, doc_scores

    # modified on 10/01/2023
    # modified on 2/9/2024, add metadata
    # modified on 2/15/2024, add keywords
    def _update_from_documents_by_batch(
        self, documents, metadata, keywords, visualize=False
    ):
        """
        This function will help update/construct a database from raw text

        Args:
            documents (List[str]): input documents
            visualize (bool, optional): Whether to visualize the progress or not. Defaults to False.
        """

        # calculate document embeddings
        if "doc" in self.config.retrieval_type:
            start_time = time.time()
            doc_embeddings = self._encoding_batch_parallel(
                documents, self.inference_urls["sentence_encoding"]
            )
            print(
                "document embeddings are calculated in %.1f s"
                % (time.time() - start_time)
            )
        else:
            doc_embeddings = [""] * len(documents)

        # calculate keywords embeddings
        # assume that all the keywords in this batch are empty if the first keyword is empty
        if keywords[0] != "":
            start_time = time.time()
            keywords_embeddings = self._encoding_batch_parallel(
                keywords, self.inference_urls["sentence_encoding"]
            )
            print(
                "keywords embeddings are calculated in %.1f s"
                % (time.time() - start_time)
            )
        else:
            keywords_embeddings = [""] * len(keywords)

        # update the document and document_neighbor_mapping databases and get document ids
        start_time = time.time()
        document_ids = self.knowledge_engine_connection.insert_documents(
            documents,
            doc_embeddings,
            metadata,
            keywords,
            keywords_embeddings,
            return_dids=True,
        )
        # document_ids = self.knowledge_engine_connection.insert_documents(documents, return_dids=True)
        print(
            "documents are inserted into databases in %.1f s"
            % (time.time() - start_time)
        )

        # parse the documents into propositions
        if (
            "prop" in self.config.retrieval_type
            or "concept" in self.config.retrieval_type
        ):
            start_time = time.time()
            propositions = self._proposition_generation_batch(input_queries=documents)
            propositions_without_none = list()
            for i, tmp_propositions in enumerate(propositions):
                if tmp_propositions is None:
                    propositions_without_none.append([documents[i]])
                else:
                    propositions_without_none.append(list(set(tmp_propositions)))
            print(
                "documents are parsed into propositions in %.1f s"
                % (time.time() - start_time)
            )

            # update the document proposition mapping database
            start_time = time.time()
            document_proposition_mappings = []
            for doc_pos, tmp_document_id in enumerate(document_ids):
                propositions = propositions_without_none[doc_pos]
                tmp_proposition_ids = (
                    self.knowledge_engine_connection.get_proposition_ids(
                        propositions=propositions
                    )
                )
                for tmp_proposition_id in tmp_proposition_ids:
                    document_proposition_mappings.append(
                        (tmp_document_id, tmp_proposition_id)
                    )
            self.knowledge_engine_connection.insert_document_proposition_mappings(
                document_proposition_mappings=document_proposition_mappings
            )
            print(
                "document-proposition mappings are inserted into databases in %.1f s"
                % (time.time() - start_time)
            )

            # parse the propositions into concepts and perspectives
            start_time = time.time()
            all_propositions = []
            for tmp_propositions in propositions_without_none:
                all_propositions += tmp_propositions
            all_propositions = list(set(all_propositions))
            filtered_propositions = (
                self.knowledge_engine_connection.filter_propositions(
                    propositions=all_propositions
                )
            )
            all_propositions = filtered_propositions

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures_key_concept_and_perspectives = executor.submit(
                    self._concept_perspective_generation_batch, all_propositions
                )
                futures_proposition_embeddings = executor.submit(
                    self._encoding_batch_parallel,
                    all_propositions,
                    self.inference_urls["sentence_encoding"],
                )
                futures_concepts = executor.submit(
                    self._concept_identification_batch, all_propositions
                )

                key_concept_and_perspectives = (
                    futures_key_concept_and_perspectives.result()
                )
                proposition_embeddings = futures_proposition_embeddings.result()
                concepts = futures_concepts.result()

                # added on 11/13/2023
                concept_perspective_concat = []
                for i in key_concept_and_perspectives:
                    if (
                        i is None
                        or not isinstance(i["concept"], str)
                        or not isinstance(i["perspective"], str)
                    ):
                        concept_perspective_concat.append("")
                    else:
                        concept_perspective_concat.append(
                            i["concept"] + " " + i["perspective"]
                        )
                futures_concept_perspective_embeddings = executor.submit(
                    self._encoding_batch_parallel,
                    concept_perspective_concat,
                    self.inference_urls["sentence_encoding"],
                )
                concept_perspective_embeddings = (
                    futures_concept_perspective_embeddings.result()
                )

                concepts_without_none = list()
                for tmp_concepts in concepts:
                    if tmp_concepts is not None:
                        concepts_without_none.append(list(set(tmp_concepts)))
                    else:
                        concepts_without_none.append([])
                concepts = concepts_without_none

            print(
                "propositions are parsed into concepts, perspectives and embeddings in %.1f s"
                % (time.time() - start_time)
            )

            # update the proposition database
            start_time = time.time()
            propositions_with_all_info = []
            for i, tmp_proposition in enumerate(all_propositions):
                if key_concept_and_perspectives[i] is None:
                    propositions_with_all_info.append([tmp_proposition, "", "", "", ""])
                else:
                    try:
                        assert isinstance(
                            key_concept_and_perspectives[i]["concept"], str
                        )
                        assert isinstance(
                            key_concept_and_perspectives[i]["perspective"], str
                        )
                        # propositions_with_all_info.append([tmp_proposition, key_concept_and_perspectives[i]['concept'], key_concept_and_perspectives[i]['perspective'], proposition_embeddings[i]])
                        propositions_with_all_info.append(
                            [
                                tmp_proposition,
                                key_concept_and_perspectives[i]["concept"],
                                key_concept_and_perspectives[i]["perspective"],
                                proposition_embeddings[i],
                                concept_perspective_embeddings[i],
                            ]
                        )
                    except AssertionError:
                        propositions_with_all_info.append(
                            [tmp_proposition, "", "", "", ""]
                        )
            proposition_ids = self.knowledge_engine_connection.insert_propositions(
                propositions_with_all_info=propositions_with_all_info, return_pids=True
            )
            print(
                "propositions are inserted into databases in %.1f s"
                % (time.time() - start_time)
            )

            # update the concept database and get concept ids
            start_time = time.time()
            all_concepts = []
            for tmp_concepts in concepts:
                if tmp_concepts is None:
                    continue
                for tmp_c in tmp_concepts:
                    if isinstance(tmp_c, str):
                        all_concepts.append(tmp_c)
            all_concepts = list(set(all_concepts))
            new_concepts = self.knowledge_engine_connection.filter_concepts(
                concepts=all_concepts
            )
            concepts_with_desc_and_info = []
            for tmp_concept in new_concepts:
                concepts_with_desc_and_info.append([tmp_concept, None, None])
            self.knowledge_engine_connection.insert_concepts(
                concept_with_desc_and_info=concepts_with_desc_and_info,
                return_cids=False,
            )
            print(
                "concepts are inserted into databases in %.1f s"
                % (time.time() - start_time)
            )

            # update the proposition concept mapping database
            start_time = time.time()
            proposition_concept_mappings = []
            for pro_pos, tmp_proposition_id in enumerate(proposition_ids):
                tmp_concepts = concepts[pro_pos]
                tmp_concept_ids = self.knowledge_engine_connection.get_concept_ids(
                    concepts=tmp_concepts
                )
                for tmp_concept_id in tmp_concept_ids:
                    proposition_concept_mappings.append(
                        [tmp_proposition_id, tmp_concept_id]
                    )
            self.knowledge_engine_connection.insert_proposition_concept_mappings(
                proposition_concept_mappings=proposition_concept_mappings
            )
            print(
                "proposition-concept mappings are inserted into databases in %.1f s"
                % (time.time() - start_time)
            )

    # modified on 2/9/2024, add metadata function
    # modified on 2/15/2024, add keywords function
    def update_from_documents(
        self, documents, metadata=None, keywords=None, batch_size=1024, visualize=False
    ):
        """
        This function will help update/construct a database from input documents
        :param keywords: keywords for documents
        :param metadata: metadata for documents
        :param documents: documents to construct from
        :type documents: list[str]
        :param batch_size: control the batch_size to avoid memory error
        :type batch_size: int
        :param visualize: Whether to visualize the progress or not
        :type visualize: bool
        :return: None
        """
        if metadata is None:
            metadata = [""] * len(documents)
        assert len(metadata) == len(documents)
        if keywords is None:
            keywords = [""] * len(documents)
        assert len(keywords) == len(documents)

        num_batch = int(len(documents) / batch_size)
        if num_batch * batch_size < len(documents):
            num_batch += 1
        if visualize:
            progress_bar = tqdm(range(num_batch), desc="Updating from documents")
        else:
            progress_bar = range(num_batch)
        for i in progress_bar:
            print("\nbatch:", i + 1, "/", num_batch)
            tmp_documents = documents[i * batch_size : (i + 1) * batch_size]
            tmp_metadata = metadata[i * batch_size : (i + 1) * batch_size]
            tmp_keywords = keywords[i * batch_size : (i + 1) * batch_size]
            self._update_from_documents_by_batch(
                documents=tmp_documents,
                metadata=tmp_metadata,
                keywords=tmp_keywords,
                visualize=visualize,
            )
        if self.mode == "inference":
            self.setup_retriever()
        self.show_statistics()

    def _updating_embedding(self, ids):
        """This function will update the embeddings of propositions with given ids

        Args:
            ids (List[str]): ids of propositions
        """
        existing_rows = self.db_connection.get_rows_by_single_key_multiple_values(
            table_name=PROPOSITION_TABLE_NAME,
            by="_id",
            keys=ids,
            columns=PROPOSITION_COLUMNS,
        )
        embeddings = self._encoding_batch_parallel(
            queries=[r["content"] for r in existing_rows],
            urls=self.inference_urls["sentence_encoding"],
        )
        for i, tmp_embedding in enumerate(embeddings):
            existing_rows[i]["embedding"] = tmp_embedding
        update_ops = self.db_connection.get_update_op(
            update_columns=["embedding"], operator="="
        )
        self.db_connection.update_rows(
            table_name=PROPOSITION_TABLE_NAME,
            rows=existing_rows,
            update_ops=update_ops,
            update_columns=["embedding"],
        )

    # added on 11/16/2023
    def _updating_cp_embedding(self, ids):
        existing_rows = self.db_connection.get_rows_by_single_key_multiple_values(
            table_name=PROPOSITION_TABLE_NAME,
            by="_id",
            keys=ids,
            columns=PROPOSITION_COLUMNS,
        )
        embeddings = self._encoding_batch_parallel(
            queries=[
                r["key_concept"] + " " + r["key_perspective"] for r in existing_rows
            ],
            urls=self.inference_urls["sentence_encoding"],
        )
        for i, tmp_embedding in enumerate(embeddings):
            existing_rows[i]["concept_perspective_embedding"] = tmp_embedding
        update_ops = self.db_connection.get_update_op(
            update_columns=["concept_perspective_embedding"], operator="="
        )
        self.db_connection.update_rows(
            table_name=PROPOSITION_TABLE_NAME,
            rows=existing_rows,
            update_ops=update_ops,
            update_columns=["concept_perspective_embedding"],
        )

    # added on 11/16/2023
    def _updating_doc_embedding(self, ids):
        existing_rows = self.db_connection.get_rows_by_single_key_multiple_values(
            table_name=DOCUMENT_TABLE_NAME, by="_id", keys=ids, columns=DOCUMENT_COLUMNS
        )
        embeddings = self._encoding_batch_parallel(
            queries=[r["content"] for r in existing_rows],
            urls=self.inference_urls["sentence_encoding"],
        )
        for i, tmp_embedding in enumerate(embeddings):
            existing_rows[i]["embedding"] = tmp_embedding
        update_ops = self.db_connection.get_update_op(
            update_columns=["embedding"], operator="="
        )
        self.db_connection.update_rows(
            table_name=DOCUMENT_TABLE_NAME,
            rows=existing_rows,
            update_ops=update_ops,
            update_columns=["embedding"],
        )

    # added on 2/15/2024
    def _updating_keywords_embedding(self, ids):
        existing_rows = self.db_connection.get_rows_by_single_key_multiple_values(
            table_name=DOCUMENT_TABLE_NAME, by="_id", keys=ids, columns=DOCUMENT_COLUMNS
        )
        embeddings = self._encoding_batch_parallel(
            queries=[r["keywords"] for r in existing_rows],
            urls=self.inference_urls["sentence_encoding"],
        )
        for i, tmp_embedding in enumerate(embeddings):
            existing_rows[i]["keywords_embedding"] = tmp_embedding
        update_ops = self.db_connection.get_update_op(
            update_columns=["keywords_embedding"], operator="="
        )
        self.db_connection.update_rows(
            table_name=DOCUMENT_TABLE_NAME,
            rows=existing_rows,
            update_ops=update_ops,
            update_columns=["keywords_embedding"],
        )

    def create_index(self, table_name, column, index_name=None):
        """
        This function will build index for given column
        :param table_name: table name
        :type table_name: str
        :param column: column name
        :type column: str
        :param index_name: index name
        :type index_name: str
        :return: None
        """
        if index_name is None:
            index_name = table_name + "_" + column + "_index"
        print(
            "Creating index for table {} column {} with index name {}".format(
                table_name, column, index_name
            )
        )
        self.db_connection.create_index(
            table_name=table_name, column=column, index_name=index_name
        )

        # Supporting Inference Functions

    def _concept_perspective_generation_batch(self, input_queries):
        """
        This function is used to identify the key concept and query from the input query.
        :param input_queries: The input queries
        :type input_queries: list[str]
        :return:
        """

        model_inputs = list()
        for tmp_query in input_queries:
            model_inputs.append(
                "<CPG>请找出问题'" + tmp_query + "'里的关键概念和角度</CPG>"
            )
        inference_results = call_tgi_server_batch_parallel(
            queries=model_inputs,
            urls=self.inference_urls["concept_perspective_generation"],
        )
        resutls_for_return = list()
        for tmp_instance in inference_results:
            try:
                concept_perspective = ast.literal_eval(tmp_instance.strip())
                assert "concept" in concept_perspective
                assert "perspective" in concept_perspective
                resutls_for_return.append(concept_perspective)
            except:
                resutls_for_return.append(None)
        return resutls_for_return

    def _proposition_generation_batch(self, input_queries):
        """
        This function is used to extract key propositions from the input queries
        :param input_queries: The input queries
        :type input_queries: list[str]
        :return:
        """
        model_inputs = list()
        for tmp_query in input_queries:
            model_inputs.append(
                "将“"
                + tmp_query
                + '”拆解成更细粒度的短句，每个短句应该包括完整的信息。简单来说，一个短句就是一个你能够通过查询来证明，证伪或回答的点。所有短句都应该是基于原句的改写，不要创造内容。要保证每个语义单元信息完整，语法通顺。每个短句中不应包含指代词。缺失的指代词要补成对应的实体或事件。要保证每个短句不包含分号等排比结构以及"和"，"或"等连接词。以json格式输出。以列表输出。'
            )
        inference_results = call_tgi_server_batch_parallel(
            queries=model_inputs, urls=self.inference_urls["proposition_generation"]
        )

        results_for_return = []
        for tmp_instance in inference_results:
            try:
                concept_perspective = ast.literal_eval(tmp_instance.strip())
                assert isinstance(concept_perspective, list)
                results_for_return.append(concept_perspective)
            except:
                results_for_return.append(None)
        return results_for_return

    def _concept_identification_batch(self, input_queries):
        """
        This function is used to identify all concepts from the input query.
        :param input_queries: The input queries
        :type input_queries: list[str]
        :return:
        """

        model_inputs = list()
        for tmp_query in input_queries:
            model_inputs.append("请找出问题“" + tmp_query + "'里的概念。")
        inference_results = call_tgi_server_batch_parallel(
            queries=model_inputs, urls=self.inference_urls["concept_identification"]
        )
        resutls_for_return = list()
        for tmp_instance in inference_results:
            try:
                concepts = ast.literal_eval(tmp_instance.strip())
                assert isinstance(concepts, list)
                resutls_for_return.append(concepts)
            except:
                resutls_for_return.append(None)
        return resutls_for_return

    # added on 1/9/2024
    # modified on 2/14/2024, new prompt
    def filter_doc_batch(self, queries, docs):
        model_inputs = []
        for q, d in zip(queries, docs):
            model_inputs.append(
                "请判断以下文档：“"
                + d
                + "”是否包含与以下查询：“"
                + q
                + "”相关的信息。是的话输出“是”，否则输出“否”。"
            )
        results = call_tgi_server_batch_parallel(
            queries=model_inputs, urls=self.inference_urls["filter_doc"]
        )
        return results

    # added on 5/30/2024
    def dialog_summarization_batch(self, input_queries):
        model_inputs = []
        for query in input_queries:
            model_inputs.append(
                "总结以下用户和助手之间的对话并给出摘要，要保留关键的信息：" + query
            )
        results = call_tgi_server_batch_parallel(
            queries=model_inputs, urls=self.inference_urls["dialog_summarization"]
        )
        return [i.strip() for i in results]

    # modified on 10/24/2023
    # modified on 2/15/2024, add keyword
    def setup_retriever(self):
        """
        This function will setup/update the retriever with the proposition embeddings.
        """
        start_time = time.time()

        # modified on 3/7/2024
        if "prop" in self.config.retrieval_type:
            all_proposition_ids = []
            all_proposition_embeddings = []
            returned_rows = self.db_connection.get_columns(
                PROPOSITION_TABLE_NAME, ["_id", "embedding"]
            )
            invalid_ids = []
            for tmp_row in returned_rows:
                try:
                    all_proposition_embeddings.append(json.loads(tmp_row["embedding"]))
                    all_proposition_ids.append(tmp_row["_id"])
                except json.JSONDecodeError:
                    invalid_ids.append(tmp_row["_id"])
            while len(invalid_ids) > 0:
                print("prop embedding invalid id num: %d" % len(invalid_ids))
                self._updating_embedding(ids=invalid_ids)
                all_proposition_ids = []
                all_proposition_embeddings = []
                returned_rows = self.db_connection.get_columns(
                    PROPOSITION_TABLE_NAME, ["_id", "embedding"]
                )
                invalid_ids = []
                for tmp_row in tqdm(returned_rows):
                    try:
                        all_proposition_embeddings.append(
                            json.loads(tmp_row["embedding"])
                        )
                        all_proposition_ids.append(tmp_row["_id"])
                    except json.JSONDecodeError:
                        invalid_ids.append(tmp_row["_id"])
            all_proposition_embeddings = np.array(
                all_proposition_embeddings, dtype=np.float16
            )
            self.retriever = KR(
                sentences=all_proposition_ids, embeddings=all_proposition_embeddings
            )

        # modified on 3/7/2024
        if "concept" in self.config.retrieval_type:
            # added on 11/13/2023 for concept and perspective embedding
            all_proposition_ids = []
            all_cp_embeddings = []
            returned_rows_cp = self.db_connection.get_columns(
                PROPOSITION_TABLE_NAME, ["_id", "concept_perspective_embedding"]
            )
            invalid_ids = []
            for row in returned_rows_cp:
                try:
                    all_cp_embeddings.append(
                        json.loads(row["concept_perspective_embedding"])
                    )
                    all_proposition_ids.append(row["_id"])
                except json.JSONDecodeError:
                    invalid_ids.append(row["_id"])
            while len(invalid_ids) > 0:
                print(
                    "prop concept and perspective embedding invalid id num: %d"
                    % len(invalid_ids)
                )
                self._updating_cp_embedding(ids=invalid_ids)
                all_proposition_ids = []
                all_cp_embeddings = []
                returned_rows_cp = self.db_connection.get_columns(
                    PROPOSITION_TABLE_NAME, ["_id", "concept_perspective_embedding"]
                )
                invalid_ids = []
                for row in tqdm(returned_rows_cp):
                    try:
                        all_cp_embeddings.append(
                            json.loads(row["concept_perspective_embedding"])
                        )
                        all_proposition_ids.append(row["_id"])
                    except json.JSONDecodeError:
                        invalid_ids.append(row["_id"])
            all_cp_embeddings = np.array(all_cp_embeddings, dtype=np.float64)
            self.retriever_cp = KR(
                sentences=all_proposition_ids, embeddings=all_cp_embeddings
            )

        # modified on 3/7/2024
        if "doc" in self.config.retrieval_type:
            # added on 10/24/2023 for document embedding
            all_document_ids = []
            all_document_embeddings = []
            returned_rows_doc = self.db_connection.get_columns(
                DOCUMENT_TABLE_NAME, ["_id", "embedding"]
            )
            invalid_ids = []
            for row in returned_rows_doc:
                try:
                    all_document_embeddings.append(json.loads(row["embedding"]))
                    all_document_ids.append(row["_id"])
                except json.JSONDecodeError:
                    invalid_ids.append(row["_id"])
            while len(invalid_ids) > 0:
                print("doc embedding invalid id num: %d" % len(invalid_ids))
                self._updating_doc_embedding(ids=invalid_ids)
                all_document_ids = []
                all_document_embeddings = []
                returned_rows_doc = self.db_connection.get_columns(
                    DOCUMENT_TABLE_NAME, ["_id", "embedding"]
                )
                invalid_ids = []
                for row in tqdm(returned_rows_doc):
                    try:
                        all_document_embeddings.append(json.loads(row["embedding"]))
                        all_document_ids.append(row["_id"])
                    except json.JSONDecodeError:
                        invalid_ids.append(row["_id"])
            all_document_embeddings = np.array(
                all_document_embeddings, dtype=np.float64
            )
            self.retriever_doc = KR(
                sentences=all_document_ids, embeddings=all_document_embeddings
            )

        # modified on 3/7/2024
        if "keyword" in self.config.retrieval_type:
            # added on 2/15/2024 for keyword embedding
            all_document_ids = []
            all_keywords_embeddings = []
            returned_rows_doc = self.db_connection.get_columns(
                DOCUMENT_TABLE_NAME, ["_id", "keywords_embedding"]
            )
            invalid_ids = []
            for row in returned_rows_doc:
                try:
                    all_keywords_embeddings.append(
                        json.loads(row["keywords_embedding"])
                    )
                    all_document_ids.append(row["_id"])
                except json.JSONDecodeError:
                    invalid_ids.append(row["_id"])
            while len(invalid_ids) > 0:
                print("keywords embedding invalid id num: %d" % len(invalid_ids))
                self._updating_keywords_embedding(ids=invalid_ids)
                all_document_ids = []
                all_keywords_embeddings = []
                returned_rows_doc = self.db_connection.get_columns(
                    DOCUMENT_TABLE_NAME, ["_id", "keywords_embedding"]
                )
                invalid_ids = []
                for row in tqdm(returned_rows_doc):
                    try:
                        all_keywords_embeddings.append(
                            json.loads(row["keywords_embedding"])
                        )
                        all_document_ids.append(row["_id"])
                    except json.JSONDecodeError:
                        invalid_ids.append(row["_id"])
            all_keywords_embeddings = np.array(
                all_keywords_embeddings, dtype=np.float64
            )
            self.retriever_keywords = KR(
                sentences=all_document_ids, embeddings=all_keywords_embeddings
            )

        print(
            "finish setting up retrievers in %.1f seconds." % (time.time() - start_time)
        )

    def show_statistics(self):
        """
        This function will show the statistics of the current knowledge engine module.
        """
        statistics = self.knowledge_engine_connection.show_statistics()
        print("\nStatistics:")
        for table_name in statistics:
            print("(%s, %d)" % (table_name[0], statistics[table_name][0][0]))
        print()

    @staticmethod
    def _encoding_batch(url, queries):
        """
        Encoding a batch of input queries
        :param queries: the queries to be encoded
        :return: vector
        """
        api_query = {"input": queries}
        proxies = {
            "http": None,
            "https": None,
        }
        url = "http://" + url
        x = requests.post(
            url,
            data=json.dumps(api_query),
            headers={"Content-Type": "application/json"},
            proxies=proxies,
        )
        try:
            embedding_in_list_format = json.loads(x.text)["embedding"]
            embedding_in_str_format = map(json.dumps, embedding_in_list_format)
            embedding_in_str_format = list(embedding_in_str_format)

            return embedding_in_str_format
        except json.JSONDecodeError:
            print("KnowledgeEngineModule::_encoding_batch Error:", x.text)
            return None

    def _encoding_batch_parallel(self, queries, urls):
        """
        Encoding a batch of input queries
        :param queries: the queries to be encoded
        :return: vector
        """
        queries_by_iteration = list()
        num_per_iteration = len(self.inference_urls["sentence_encoding"]) * 512
        num_iterations = int(
            len(queries) / len(self.inference_urls["sentence_encoding"])
        )
        if num_iterations * len(self.inference_urls["sentence_encoding"]) < len(
            queries
        ):
            num_iterations += 1
        for tmp_iteration in range(num_iterations):
            tmp_queries = queries[
                tmp_iteration
                * num_per_iteration : (tmp_iteration + 1)
                * num_per_iteration
            ]
            if len(tmp_queries) > 0:
                queries_by_iteration.append(tmp_queries)
        results = []
        for tmp_iteration_query in queries_by_iteration:
            num_batches = int(len(tmp_iteration_query) / 512)
            if num_batches * 512 < len(tmp_iteration_query):
                num_batches += 1
            queries_by_batch = list()
            for tmp_batch_id in range(num_batches):
                tmp_queries = tmp_iteration_query[
                    tmp_batch_id * 512 : (tmp_batch_id + 1) * 512
                ]
                queries_by_batch.append(tmp_queries)

            tmp_results = []
            with ThreadPoolExecutor(max_workers=len(queries_by_batch)) as executor:
                # use cycle to make sure the urls are used in a round-robin fashion and use map to distribute the queries.
                url_cycle = itertools.cycle(urls)
                for batch_results in executor.map(
                    self._encoding_batch, url_cycle, queries_by_batch
                ):
                    tmp_results += batch_results
            results += tmp_results

        return results


def _call_tgi_server_single_query(url, query):
    headers = {"Content-Type": "application/json"}
    data = {"inputs": query, "parameters": {"max_new_tokens": 1024}}
    url = "http://" + url + "/generate"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["generated_text"]
    elif response.status_code == 422:
        print("Failed:", response.status_code)
        return (
            "The input is too long. Please clean your history or try a shorter input."
        )
    else:
        print("Failed:", response.status_code)
        print(response.text)
        return "Failed:" + str(response.status_code)


# modified on 10/01/2023
def _call_tgi_server_batch(url, queries):
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(_call_tgi_server_single_query, url, query): query
            for query in queries
        }
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{query} generated an exception: {exc}')
                results.append(f'Error: {exc}')
    return results
    """

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in executor.map(
            _call_tgi_server_single_query, [url] * len(queries), queries
        ):
            results.append(result)
    return results


def call_tgi_server_batch_parallel(queries, urls, batch_size=64):
    queries_by_iteration = list()
    num_per_iteration = len(urls) * batch_size
    num_iterations = int(len(queries) / num_per_iteration)
    if num_iterations * num_per_iteration < len(queries):
        num_iterations += 1
    for tmp_iteration in range(num_iterations):
        tmp_queries = queries[
            tmp_iteration * num_per_iteration : (tmp_iteration + 1) * num_per_iteration
        ]
        if len(tmp_queries) > 0:
            queries_by_iteration.append(tmp_queries)
    results = []
    for tmp_iteration_query in queries_by_iteration:
        new_batch_size = min(batch_size, int(len(tmp_iteration_query) / len(urls)) + 1)
        num_batches = int(len(tmp_iteration_query) / new_batch_size) + 1
        queries_by_batch = list()
        all_number = 0
        for tmp_batch_id in range(num_batches):
            tmp_queries = tmp_iteration_query[
                tmp_batch_id * new_batch_size : (tmp_batch_id + 1) * new_batch_size
            ]
            queries_by_batch.append(tmp_queries)
            all_number += len(tmp_queries)
        assert all_number == len(tmp_iteration_query)

        tmp_results = []
        with ThreadPoolExecutor(max_workers=len(queries_by_batch)) as executor:
            # use cycle to make sure the urls are used in a round-robin fashion and use map to distribute the queries.
            url_cycle = itertools.cycle(urls)
            for batch_results in executor.map(
                _call_tgi_server_batch, url_cycle, queries_by_batch
            ):
                tmp_results += batch_results
        results += tmp_results

    return results
