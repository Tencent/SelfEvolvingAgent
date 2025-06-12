import hashlib
from collections import OrderedDict
from cognitive_kernel.memory_kernel.db_connection import SqliteDBConnection
from datetime import datetime
from collections import defaultdict


CHUNKSIZE = 32768
RELATION_SENSES = [
    "Precedence", "Succession", "Synchronous",
    "Reason", "Result",
    "Condition", "Contrast", "Concession",
    "Conjunction", "Instantiation", "Restatement",
    "ChosenAlternative", "Alternative", "Exception",
    "Co_Occurrence"
]

DOCUMENT_TABLE_NAME = "Documents"
# DOCUMENT_COLUMNS = ["_id", "content"]
# DOCUMENT_COLUMN_TYPES = ["PRIMARY KEY", "TEXT"]
# DOCUMENT_COLUMNS = ["_id", "content", "embedding"]
# DOCUMENT_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "BLOB"]
DOCUMENT_COLUMNS = ["_id", "content", "embedding", "metadata", "keywords", "keywords_embedding"]
DOCUMENT_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "BLOB", "TEXT", "TEXT", "BLOB"]

PROPOSITION_TABLE_NAME = "Propositions"
# PROPOSITION_COLUMNS = ["_id", "content", "key_concept", "key_perspective", "embedding"]
# PROPOSITION_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT", "TEXT", "BLOB"]
PROPOSITION_COLUMNS = ["_id", "content", "key_concept", "key_perspective", "embedding", "concept_perspective_embedding"]
PROPOSITION_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT", "TEXT", "BLOB", "BLOB"]

DOCUMENT_PROPOSITION_MAPPING_TABLE_NAME = "DocumentPropositionsMapping"
DOCUMENT_PROPOSITION_MAPPING_COLUMNS = ["_id", "doc_id", "prop_id"]
DOCUMENT_PROPOSITION_MAPPING_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"]

PROPOSITION_CONCEPT_MAPPING_TABLE_NAME = "PropositionConceptMapping"
PROPOSITION_CONCEPT_MAPPING_COLUMNS = ["_id", "prop_id", "concept_id"]
PROPOSITION_CONCEPT_MAPPING_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"]

PROPOSITION_RELATION_TABLE_NAME = "PropositionRelations"
PROPOSITION_RELATION_COLUMNS = ["_id", "head_prop_id", "tail_prop_id"] + RELATION_SENSES
PROPOSITION_RELATION_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"] + ["REAL"] * len(RELATION_SENSES)

CONCEPT_TABLE_NAME = "Concepts"
CONCEPT_COLUMNS = ["_id", "name", "desc", "info"]
CONCEPT_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT", "TEXT"]

CONCEPT_ABSTRACTION_MAPPING_TABLE_NAME = "ConceptAbstractionMapping"
CONCEPT_ABSTRACTION_MAPPING_COLUMNS = ["_id", "instantiation_concept", "abstraction_concept", "score"]
CONCEPT_ABSTRACTION_MAPPING_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT", "REAL"]

MENTION_CONCEPT_MAPPING_TABLE_NAME = 'MentionConceptMapping'
MENTION_CONCEPT_MAPPING_COLUMNS = ["_id", "mention", "concept_id"]
MENTION_CONCEPT_MAPPING_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"]


class KnowledgeEngineConnection(object):
    def __init__(self, db_path, db="sqlite", mode="cache", chunksize=CHUNKSIZE):
        """
        :param db_path: database path
        :type db_path: str
        :param db: the backend database, e.g., "sqlite" or "mongodb"
        :type db: str (default = "sqlite")
        :param mode: the mode to use the connection.
            "insert": this connection is only used to insert/update rows;
            "cache": this connection caches some contents that have been retrieved;
            "memory": this connection loads all contents in memory;
        :type mode: str (default = "cache")
        :param grain: the grain to build cache
            "words": cache is built on "verbs", "skeleton_words", and "words"
            "skeleton_words": cache is built on "verbs", and "skeleton_words"
            "verbs": cache is built on "verbs"
            None: no cache
        :type grain: Union[str, None] (default = None)
        :param chunksize: the chunksize to load/write database
        :type chunksize: int (default = 32768)
        """

        if db == "sqlite":
            self._conn = SqliteDBConnection(db_path, chunksize)
        elif db == "mongodb":
            raise NotImplementedError
        else:
            raise ValueError("Error: %s database is not supported!" % db)
        self.mode = mode
        if self.mode not in ["insert", "cache", "memory"]:
            raise ValueError("only support insert/cache/memory modes.")

        self.document_table_name = DOCUMENT_TABLE_NAME
        self.document_columns = DOCUMENT_COLUMNS
        self.document_column_types = DOCUMENT_COLUMN_TYPES
        self.proposition_table_name = PROPOSITION_TABLE_NAME
        self.proposition_columns = PROPOSITION_COLUMNS
        self.proposition_column_types = PROPOSITION_COLUMN_TYPES
        self.document_proposition_mapping_table_name = DOCUMENT_PROPOSITION_MAPPING_TABLE_NAME
        self.document_proposition_mapping_columns = DOCUMENT_PROPOSITION_MAPPING_COLUMNS
        self.document_proposition_mapping_column_types = DOCUMENT_PROPOSITION_MAPPING_COLUMN_TYPES
        self.proposition_concept_mapping_table_name = PROPOSITION_CONCEPT_MAPPING_TABLE_NAME
        self.proposition_concept_mapping_columns = PROPOSITION_CONCEPT_MAPPING_COLUMNS
        self.proposition_concept_mapping_column_types = PROPOSITION_CONCEPT_MAPPING_COLUMN_TYPES
        self.proposition_relation_table_name = PROPOSITION_RELATION_TABLE_NAME
        self.proposition_relation_columns = PROPOSITION_RELATION_COLUMNS
        self.proposition_relation_column_types = PROPOSITION_RELATION_COLUMN_TYPES
        self.concept_table_name = CONCEPT_TABLE_NAME
        self.concept_columns = CONCEPT_COLUMNS
        self.concept_column_types = CONCEPT_COLUMN_TYPES
        self.concept_abstraction_mapping_table_name = CONCEPT_ABSTRACTION_MAPPING_TABLE_NAME
        self.concept_abstraction_mapping_columns = CONCEPT_ABSTRACTION_MAPPING_COLUMNS
        self.concept_abstraction_mapping_column_types = CONCEPT_ABSTRACTION_MAPPING_COLUMN_TYPES
        self.mention_concept_mapping_table_name = MENTION_CONCEPT_MAPPING_TABLE_NAME
        self.mention_concept_mapping_columns = MENTION_CONCEPT_MAPPING_COLUMNS
        self.mention_concept_mapping_column_types = MENTION_CONCEPT_MAPPING_COLUMN_TYPES
        self.pids = set()
        self.cids = set()
        self.init()

    def init(self):
        """ Initialize the ASERKGConnection, including creating tables, loading eids and rids, and building cache
        """

        for table_name, columns, column_types in zip(
                [self.document_table_name,
                 self.proposition_table_name,
                 self.document_proposition_mapping_table_name,
                 self.proposition_concept_mapping_table_name,
                 self.proposition_relation_table_name,
                 self.concept_table_name,
                 self.concept_abstraction_mapping_table_name,
                 self.mention_concept_mapping_table_name
                 ],
                [self.document_columns,
                 self.proposition_columns,
                 self.document_proposition_mapping_columns,
                 self.proposition_concept_mapping_columns,
                 self.proposition_relation_columns,
                 self.concept_columns,
                 self.concept_abstraction_mapping_columns,
                 self.mention_concept_mapping_columns],
                [self.document_column_types,
                 self.proposition_column_types,
                 self.document_proposition_mapping_column_types,
                 self.proposition_concept_mapping_column_types,
                 self.proposition_relation_column_types,
                 self.concept_column_types,
                 self.concept_abstraction_mapping_column_types,
                 self.mention_concept_mapping_column_types]
        ):
            if len(columns) == 0 or len(column_types) == 0:
                raise ValueError("Error: %s_columns and %s_column_types must be defined" % (table_name, table_name))
            try:
                self._conn.create_table(table_name, columns, column_types)
            except:
                pass

        for p in self._conn.get_columns(self.proposition_table_name, ["_id"]):
            self.pids.add(p["_id"])
        for c in self._conn.get_columns(self.concept_table_name, ["_id"]):
            self.cids.add(c["_id"])

    def close(self):
        """ Close the ASERKGConnection safely

        """

        self._conn.close()

    def get_conn(self):
        """
        This function will return the db connection
        :return:
        """
        return self._conn

    def show_tables(self):
        """ Show all tables in the database

        :rtype: List[str]
        """

        return self._conn.return_table_list()

    def show_statistics(self):
        """ Show statistics of the database

        :rtype: Dict[str, int]
        """
        table_names = self._conn.return_table_list()
        result = dict()
        for tmp_table_name in table_names:
            result[tmp_table_name] = self._conn.return_table_number_row(table_name=tmp_table_name)

        return result

    def remove_all_tables(self):
        """ Remove all tables in the database

        """
        table_names = self._conn.return_table_list()
        for tmp_table_name in table_names:
            self._conn.drop_table(tmp_table_name)

    @staticmethod
    def _generate_document_id(relative_pos):
        current_time = str(datetime.now())
        return "D" + current_time + '$$' + relative_pos

    @staticmethod
    def _generate_proposition_id(proposition):
        return "P" + hashlib.md5(proposition.encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_document_proposition_mapping_id(document_id, proposition_id):
        return "DPM" + hashlib.md5((document_id + '$$' + proposition_id).encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_proposition_concept_mapping_id(proposition_id, concept_id):
        return "DCM" + hashlib.md5((proposition_id + '$$' + concept_id).encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_proposition_relation_id(head_proposition_id, tail_proposition_id):
        return "PR" + hashlib.md5((head_proposition_id + '$$' + tail_proposition_id).encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_concept_id(concept):
        return "C" + hashlib.md5(concept.encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_concept_abstraction_mapping_id(instantiation_concept_id, abstraction_concept_id):
        return "CAM" + hashlib.md5(
            (instantiation_concept_id + '$$' + abstraction_concept_id).encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_mention_concept_id(mention, concept_id):
        return "MC" + hashlib.md5((mention + '$$' + concept_id).encode("utf-8")).hexdigest()

    '''
    def _convert_document_to_row(self, document_with_pos):
        row = OrderedDict({"_id": self._generate_document_id(document_with_pos[1])})
        row['content'] = document_with_pos[0]
        return row
    '''

    # modified on 10/24/2023
    # modified on 2/9/2024, add metadata
    # modified on 2/15/2024, add keywords and keywords embedding
    def _convert_document_to_row(self, d):
        row = OrderedDict({"_id": self._generate_document_id(d[0])})
        row['content'] = d[1]
        row['embedding'] = d[2]
        row['metadata'] = d[3]
        row['keywords'] = d[4]
        row['keywords_embedding'] = d[5]
        return row

    '''
    def _convert_proposition_to_row(self, proposition, key_concept, key_perspective, embedding):
        row = OrderedDict({"_id": self._generate_proposition_id(proposition)})
        row['content'] = proposition
        row['key_concept'] = key_concept
        row['key_perspective'] = key_perspective
        row['embedding'] = embedding
        return row
    '''

    # modified on 11/13/2023
    def _convert_proposition_to_row(self, proposition, key_concept, key_perspective, proposition_embedding, concept_perspective_embedding):
        row = OrderedDict({"_id": self._generate_proposition_id(proposition)})
        row['content'] = proposition
        row['key_concept'] = key_concept
        row['key_perspective'] = key_perspective
        row['embedding'] = proposition_embedding
        row['concept_perspective_embedding'] = concept_perspective_embedding
        return row

    def _convert_document_proposition_mapping_to_row(self, document_id, proposition_id):
        row = OrderedDict({"_id": self._generate_document_proposition_mapping_id(document_id, proposition_id)})
        row['doc_id'] = document_id
        row['prop_id'] = proposition_id
        return row

    def _convert_proposition_concept_mapping_to_row(self, proposition_id, concept_id):
        row = OrderedDict({"_id": self._generate_proposition_concept_mapping_id(proposition_id, concept_id)})
        row['prop_id'] = proposition_id
        row['concept_id'] = concept_id
        return row

    def _convert_proposition_relation_to_row(self, head_proposition_id, tail_proposition_id, relation_senses):
        row = OrderedDict({"_id": self._generate_proposition_relation_id(head_proposition_id, tail_proposition_id)})
        row['head_prop_id'] = head_proposition_id
        row['tail_prop_id'] = tail_proposition_id
        for relation_sense in relation_senses:
            row[relation_sense] = relation_senses[relation_sense]
        return row

    def _convert_concept_to_row(self, concept, desc, info):
        row = OrderedDict({"_id": self._generate_concept_id(concept)})
        row['name'] = concept
        row['desc'] = desc
        row['info'] = info
        return row

    def _convert_concept_abstract_mapping_to_row(self, instantiation_concept_id, abstraction_concept_id, score):
        row = OrderedDict(
            {"_id": self._generate_concept_abstraction_mapping_id(instantiation_concept_id, abstraction_concept_id)})
        row['instantiation_concept'] = instantiation_concept_id
        row['abstraction_concept'] = abstraction_concept_id
        row['score'] = score
        return row

    def _convert_mention_mapping_to_row(self, mention, concept_id):
        row = OrderedDict({"_id": self._generate_mention_concept_id(mention, concept_id)})
        row['mention'] = mention
        row['concept_id'] = concept_id
        return row

    '''
    def insert_documents(self, documents, return_dids=False):
        """ Insert documents into the database

        :param return_dids: whether document ids is returned
        :param documents: the documents to insert
        :type documents: List[str]
        """
        documents_with_positions = list()
        for i, tmp_document in enumerate(documents):
            documents_with_positions.append((tmp_document, str(i)))
        rows = list(map(self._convert_document_to_row, documents_with_positions))
        self._conn.insert_rows(self.document_table_name, rows)
        if return_dids:
            return list(map(lambda x: x['_id'], rows))
    '''

    # modified on 10/24/2023
    # modified on 2/9/2024, add metadata
    # modified on 2/15/2024, add keywords
    def insert_documents(self, documents, doc_embeddings, metadata, keywords, keywords_embeddings, return_dids=False):
        """ Insert documents into the database
        :param keywords: keywords for documents
        :param keywords_embeddings: keywords embeddings
        :param metadata: metadata for documents
        :param doc_embeddings: document embeddings
        :param documents: documents
        :param return_dids: whether document ids is returned
        """
        documents_with_positions = []
        for i, (d, d_emb, md, k, k_emb) in enumerate(
                zip(documents, doc_embeddings, metadata, keywords, keywords_embeddings)):
            documents_with_positions.append((str(i), d, d_emb, md, k, k_emb))
        rows = list(map(self._convert_document_to_row, documents_with_positions))
        self._conn.insert_rows(self.document_table_name, rows)
        if return_dids:
            return list(map(lambda x: x['_id'], rows))

    '''
    def insert_propositions(self, propositions_with_all_info, return_pids=False):
        """ Insert propositions into the database

        :param return_pids:
        :param propositions_with_all_info:
        """
        rows = list(map(lambda x: self._convert_proposition_to_row(x[0], x[1], x[2], x[3]), propositions_with_all_info))
        self._conn.insert_rows(self.proposition_table_name, rows)
        # adding pids
        for tmp_row in rows:
            self.pids.add(tmp_row['_id'])
        if return_pids:
            return list(map(lambda x: x['_id'], rows))
    '''

    # modified on 11/13/2023
    def insert_propositions(self, propositions_with_all_info, return_pids=False):
        """ Insert propositions into the database

        :param return_pids:
        :param propositions_with_all_info:
        """
        rows = list(
            map(lambda x: self._convert_proposition_to_row(x[0], x[1], x[2], x[3], x[4]), propositions_with_all_info))
        self._conn.insert_rows(self.proposition_table_name, rows)
        # adding pids
        for tmp_row in rows:
            self.pids.add(tmp_row['_id'])
        if return_pids:
            return list(map(lambda x: x['_id'], rows))

    def insert_document_proposition_mappings(self, document_proposition_mappings):
        """ Insert document proposition mappings into the database

        :param document_proposition_mappings: the document proposition mappings to insert
        :type document_proposition_mappings: List[Tuple[str, str]]
        """
        rows = list(
            map(lambda x: self._convert_document_proposition_mapping_to_row(x[0], x[1]), document_proposition_mappings))
        self._conn.insert_rows(self.document_proposition_mapping_table_name, rows)

    def insert_proposition_concept_mappings(self, proposition_concept_mappings):
        """ Insert proposition concept mappings into the database

        :param proposition_concept_mappings: the proposition concept mappings to insert
        :type proposition_concept_mappings: List[Tuple[str, str]]
        """
        # tmp_start_time = time.time()
        rows = list(
            map(lambda x: self._convert_proposition_concept_mapping_to_row(x[0], x[1]), proposition_concept_mappings))
        # print('convert time: {}'.format(time.time() - tmp_start_time))
        # tmp_start_time = time.time()
        self._conn.insert_rows(self.proposition_concept_mapping_table_name, rows)
        # print('insert time: {}'.format(time.time() - tmp_start_time))
        # tmp_start_time = time.time()
        # sorted_rows = sorted(rows, key=lambda x: x['_id'])
        # self._conn.insert_rows(self.proposition_concept_mapping_table_name, sorted_rows)
        # print('insert time: {}'.format(time.time() - tmp_start_time))

    def insert_proposition_relations(self, proposition_relations):
        """ Insert proposition relations into the database

        :param proposition_relations: the proposition relations to insert
        :type proposition_relations: List[Tuple[str, str, Dict[str, float]]]
        """
        rows = list(map(lambda x: self._convert_proposition_relation_to_row(x[0], x[1], x[2]), proposition_relations))
        self._conn.insert_rows(self.proposition_relation_table_name, rows)

    def insert_concepts(self, concept_with_desc_and_info, return_cids=False):
        """ Insert concepts into the database

        :param concept_with_desc_and_info: the concepts to insert
        :type concept_with_desc_and_info: List[Tuple[str, str, str]]
        """
        rows = list(map(lambda x: self._convert_concept_to_row(x[0], x[1], x[2]), concept_with_desc_and_info))
        self._conn.insert_rows(self.concept_table_name, rows)
        for tmp_row in rows:
            self.cids.add(tmp_row['_id'])
        if return_cids:
            return list(map(lambda x: x['_id'], rows))

    def insert_concept_abstract_mappings(self, concept_abstract_mappings):
        """ Insert concept abstract mappings into the database

        :param concept_abstract_mappings: the concept abstract mappings to insert
        :type concept_abstract_mappings: List[Tuple[str, str, float]]
        """
        rows = list(
            map(lambda x: self._convert_concept_abstract_mapping_to_row(x[0], x[1], x[2]), concept_abstract_mappings))
        self._conn.insert_rows(self.concept_abstract_mapping_table_name, rows)

    def insert_mention_mappings(self, mention_mappings):
        """ Insert mention mappings into the database

        :param mention_mappings: the mention mappings to insert
        :type mention_mappings: List[Tuple[str, str]]
        """
        rows = list(map(lambda x: self._convert_mention_mapping_to_row(x[0], x[1]), mention_mappings))
        self._conn.insert_rows(self.mention_mapping_table_name, rows)

    def get_proposition_ids(self, propositions):
        """ Get proposition ids from propositions

        :param propositions: the propositions to retrieve
        :type propositions: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
        return list(map(lambda x: self._generate_proposition_id(x), propositions))

    def get_concept_ids(self, concepts):
        """ Get concept ids from concepts

        :param concepts: the concepts to retrieve
        :type concepts: List[str]
        """
        return list(map(lambda x: self._generate_concept_id(x), concepts))

    def retrieve_proposition_ids_by_concepts(self, concepts):
        """ Retrieve propositions by concepts

        :param concepts: the concepts to retrieve
        :type concepts: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
        cids = self.get_concept_ids(concepts)
        raw_result = self._conn.get_rows_by_single_key_multiple_values(
            table_name=self.proposition_concept_mapping_table_name,
            by='concept_id',
            keys=cids,
            columns=['prop_id'])
        pids = list(map(lambda x: x['prop_id'], raw_result))
        return pids

    def retrieve_documents_by_propositions(self, propositions):
        """ Retrieve documents by propositions

        :param propositions: the propositions to retrieve
        :type propositions: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
        pids = self.get_proposition_ids(propositions)
        return self.retrieve_documents_by_proposition_ids(pids=pids)

    def retrieve_documents_by_proposition_ids(self, pids):
        """ Retrieve documents by proposition ids

        :param pids: the proposition ids to retrieve
        :type pids: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
        query = 'SELECT * FROM {} WHERE _id IN (SELECT document_id FROM {} WHERE proposition_id IN ({}))'.format(
            self.document_table_name, self.document_proposition_mapping_table_name, ','.join(pids))
        return self._conn._conn.execute(query)

    def retrieve_neighbors_by_document_ids(self, dids, num_neighbors=10):
        raise NotImplementedError

    '''
    # modified on 11/06/2023 for ordered results in soft matching
    def retrieve_doc_content_based_on_proposition_ids(self, pids, neighbor_size=5):
        doc_content_list = []
        doc_id_set = set()
        for pid in pids:
            target_did_raw_data = self._conn.get_rows_by_keys(table_name=DOCUMENT_PROPOSITION_MAPPING_TABLE_NAME, bys=['prop_id'], keys=[pid], columns=['doc_id'])
            # assert len(target_did_raw_data) == 1
            # one proposition may map to multiple documents
            for i in range(len(target_did_raw_data)):
                target_did = target_did_raw_data[i]['doc_id']
                if target_did not in doc_id_set:
                    retrieved_document = self._conn.get_rows_by_keys(table_name=DOCUMENT_TABLE_NAME, bys=['_id'], keys=[target_did], columns=['_id', 'content'])
                    assert len(retrieved_document) == 1
                    content = retrieved_document[0]['content']
                    doc_content_list.append(content)
                    doc_id_set.add(target_did)

        return doc_content_list
    '''

    '''
    def retrieve_doc_content_based_on_proposition_ids(self, pids, neighbor_size=5):
        """This function will retrieve the document content based on the proposition ids.

        Args:
            proposition_ids (List[str]): the list of proposition ids

        Returns:
            List[str]: the list of document content
        """

        target_dids_raw_data = self._conn.get_rows_by_single_key_multiple_values(
            table_name=DOCUMENT_PROPOSITION_MAPPING_TABLE_NAME,
            by='prop_id',
            keys=pids,
            columns=['doc_id'])
        target_dids = list(map(lambda x: x['doc_id'], target_dids_raw_data))
        target_dids = list(set(target_dids))
        retrieved_documents = self._conn.get_rows_by_single_key_multiple_values(table_name=DOCUMENT_TABLE_NAME,
                                                                                by='_id',
                                                                                keys=target_dids,
                                                                                columns=['_id', 'content'])
        contents = list(map(lambda x: x['content'], retrieved_documents))
        return contents
    '''

    # modified on 1/30/2024, dealing with scores
    # modified on 2/9/2024, add metadata
    def retrieve_doc_content_based_on_proposition_ids(self, pids, prop_scores, neighbor_size=5):
        did2score = defaultdict(float)
        for pid, score in zip(pids, prop_scores):
            target_did_raw_data = self._conn.get_rows_by_keys(table_name=DOCUMENT_PROPOSITION_MAPPING_TABLE_NAME,
                                                              bys=['prop_id'], keys=[pid], columns=['doc_id'])
            # one proposition may map to multiple documents
            for i in range(len(target_did_raw_data)):
                target_did = target_did_raw_data[i]['doc_id']
                # prop to doc mapping strategy
                # did2score[target_did] += score
                did2score[target_did] = max(did2score[target_did], score)

        doc_content = []
        doc_metadata = []
        doc_scores = []
        for did, score in did2score.items():
            retrieved_document = self._conn.get_rows_by_keys(table_name=DOCUMENT_TABLE_NAME, bys=['_id'], keys=[did],
                                                             columns=['_id', 'content', 'metadata'])
            assert len(retrieved_document) == 1
            content = retrieved_document[0]['content']
            metadata = retrieved_document[0]['metadata']
            doc_content.append(content)
            doc_metadata.append(metadata)
            doc_scores.append(score)

        return doc_content, doc_metadata, doc_scores

    '''
    def retrieve_proposition_content_based_on_proposition_ids(self, pids):
        raw_propositions = self._conn.get_rows_by_single_key_multiple_values(table_name=PROPOSITION_TABLE_NAME,
                                                                             by='_id',
                                                                             keys=pids,
                                                                             columns=['_id', 'content'])
        return list(map(lambda x: x['content'], raw_propositions))
    '''

    # modified on 11/06/2023 for ordered results in soft matching
    def retrieve_proposition_content_based_on_proposition_ids(self, pids):
        contents = []
        for pid in pids:
            raw_proposition = self._conn.get_rows_by_keys(table_name=PROPOSITION_TABLE_NAME, bys=['_id'], keys=[pid],
                                                          columns=['_id', 'content'])
            assert len(raw_proposition) == 1
            contents.append(raw_proposition[0]['content'])
        return contents

    '''
    # added on 10/24/2023 for document dense retrieval
    def retrieve_doc_content_based_on_doc_ids(self, dids):
        raw_documents = self._conn.get_rows_by_single_key_multiple_values(table_name=DOCUMENT_TABLE_NAME,
                                                                          by='_id',
                                                                          keys=dids,
                                                                          columns=['_id', 'content'])
        return list(map(lambda x: x['content'], raw_documents))
    '''

    # modified on 11/06/2023 for ordered results in soft matching
    # modified on 2/9/2024, add metadata
    def retrieve_doc_content_based_on_doc_ids(self, dids):
        doc_contents = []
        doc_metadata = []
        for did in dids:
            doc_content = self._conn.get_rows_by_keys(table_name=DOCUMENT_TABLE_NAME, bys=['_id'], keys=[did],
                                                      columns=['_id', 'content', 'metadata'])
            assert len(doc_content) == 1
            doc_contents.append(doc_content[0]['content'])
            doc_metadata.append(doc_content[0]['metadata'])
        return doc_contents, doc_metadata

    def filter_propositions(self, propositions):
        """ Filter propositions that already exist in the database

        :param propositions: the propositions to filter
        :type propositions: List[str]
        :return: the filtered propositions
        :rtype: List[str]
        """
        return list(filter(lambda x: self._generate_proposition_id(x) not in self.pids, propositions))

    def filter_concepts(self, concepts):
        """ Filter concepts that already exist in the database

        :param concepts: the concepts to filter
        :type concepts: List[str]
        :return: the filtered concepts
        :rtype: List[str]
        """
        return list(filter(lambda x: self._generate_concept_id(x) not in self.cids, concepts))
