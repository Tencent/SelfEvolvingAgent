import os
import json
import sqlite3
from collections import defaultdict, OrderedDict


class BaseDBConnection(object):
    """ Base KG connection for database

    """
    def __init__(self, chunksize):
        """ Create an connection to database

        :param chunksize: the chunksize to load/write database
        :type chunksize: int
        """

        self._conn = None
        self.chunksize = chunksize

    def close(self):
        """ Close the connection safely
        """
        raise NotImplementedError

    def __del__(self):
        self.close()

    def create_table(self, table_name, columns, column_types):
        """ Create a table with given columns and types

        :param table_name: the table name to create
        :type table_name: str
        :param columns: the columns to create
        :type columns: List[str]
        :param column_types: the corresponding column types
        :type column_types: List[str]
        """

        raise NotImplementedError

    def get_columns(self, table_name, columns):
        """ Get column information from a table

        :param table_name: the table name to retrieve
        :type table_name: str
        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """

        raise NotImplementedError

    def select_row(self, table_name, _id, columns):
        """ Select a row from a table

        :param table_name: the table name to retrieve
        :type table_name: str
        :param _id: the row id
        :type _id: str
        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a retrieved row
        :rtype: Dict[str, object]
        """

        raise NotImplementedError

    def select_rows(self, table_name, _ids, columns):
        """ Select rows from a table

        :param table_name: the table name to retrieve
        :type table_name: str
        :param _ids: the row ids
        :type _ids: List[str]
        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: retrieved rows
        :rtype: List[Dict[str, object]]
        """

        raise NotImplementedError

    def insert_row(self, table_name, row):
        """ Insert a row into a table

        :param table_name: the table name to insert
        :type table_name: str
        :param row: the row to insert
        :type row: Dict[str, object]
        """

        raise NotImplementedError

    def insert_rows(self, table_name, rows):
        """ Insert several rows into a table

        :param table_name: the table name to insert
        :type table_name: str
        :param rows: the rows to insert
        :type rows: List[Dict[str, object]]
        """

        raise NotImplementedError

    def get_update_op(self, update_columns, operator):
        """ Get an update operator based on columns and a operator

        :param update_columns: a list of columns to update
        :type update_columns: List[str]
        :param operator: an operator that applies to the columns, including "+", "-", "*", "/", "="
        :type operator: str
        :return: an operator that suits the backend database
        :rtype: object
        """
        raise NotImplementedError

    def update_row(self, table_name, row, update_op, update_columns):
        """ Update a row that exists in a table

        :param table_name: the table name to update
        :type table_name: str
        :param row: a new row
        :type row: Dict[str, object]
        :param update_op: an operator that returned by `get_update_op`
        :type update_op: object
        :param update_columns: the columns to update
        :type update_columns: List[str]
        """

        raise NotImplementedError

    def update_rows(self, table_name, rows, update_ops, update_columns):
        """ Update rows that exist in a table

        :param table_name: the table name to update
        :type table_name: str
        :param rows: new rows
        :type rows: List[Dict[str, object]]
        :param update_ops: operator(s) that returned by `get_update_op`
        :type update_ops: Union[List[object], object]
        :param update_columns: the columns to update
        :type update_columns: List[str]
        """

        raise NotImplementedError

    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        """ Retrieve rows by specific keys in some order

        :param table_name: the table name to retrieve
        :type table_name: str
        :param bys: the given columns to match
        :type bys: List[str]
        :param keys: the given values to match
        :type keys: List[str]
        :param columns: the given columns to retrieve
        :type columns: List[str]
        :param order_bys: the columns whose value are used to sort rows
        :type order_bys: List[str]
        :param reverse: whether to sort in a reversed order
        :type reverse: bool
        :param top_n: how many rows to return, default `None` for all rows
        :type top_n: int
        :return: retrieved rows
        :rtype: List[Dict[str, object]]
        """
        raise NotImplementedError


class SqliteDBConnection(BaseDBConnection):
    """ KG connection for SQLite database
    """

    def __init__(self, db_path, chunksize):
        """ Create an connection to SQLite database

        :param db_path: database path, e.g., /home/xliucr/ASER/KG.db
        :type db_path: str
        :param chunksize: the chunksize to load/write database
        :type chunksize: int
        """

        import sqlite3
        super(SqliteDBConnection, self).__init__(chunksize)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)

    def close(self):
        """ Close the connection safely
        """
        if self._conn:
            self._conn.close()

    def quick_demo(self, table_name, columns, num_instances):
        """
        This function will show the top instances for demo purpose
        :param table_name: The table to show
        :type table_name: str
        :param columns: the columns to create
        :type columns: List[str]
        :param num_instances: Number of instances to show
        :type num_instances: int
        :return: list of columns
        """
        # num_rows = self._conn.execute('SELECT COUNT(*) FROM %s;' % table_name)
        # print('Number of rows:', num_rows.fetchone()[0])
        select_row = "SELECT %s FROM %s LIMIT %d;" % (",".join(columns), table_name, num_instances)
        result = list(map(lambda x: OrderedDict(zip(columns, x)), self._conn.execute(select_row)))
        # print(result)
        return result

    def random_sample(self, table_name, columns, num_instances):
        """
        This function will random sample a few instances from the db for demo
        :param table_name: The table to show
        :type table_name: str
        :param columns: the columns to create
        :type columns: List[str]
        :param num_instances: Number of instances to show
        :type num_instances: int
        :return: list of columns
        """
        # num_rows = self._conn.execute('SELECT COUNT(*) FROM %s;' % table_name)
        # print('Number of rows:', num_rows.fetchone()[0])

        select_row = "SELECT %s FROM %s WHERE rowid >" % (",".join(columns), table_name)
        select_row += "(abs(random()) % (SELECT max(rowid) FROM "
        select_row += table_name
        select_row += " ))"
        select_row += " LIMIT %d;" % num_instances
        result = list(map(lambda x: OrderedDict(zip(columns, x)), self._conn.execute(select_row)))
        # print(result)
        return result

    def create_table(self, table_name, columns, column_types):
        """ Create a table with given columns and types

        :param table_name: the table name to create
        :type table_name: str
        :param columns: the columns to create
        :type columns: List[str]
        :param column_types: the corresponding column types, please refer to https://www.sqlite.org/datatype3.html
        :type column_types: List[str]
        """

        create_table = "CREATE TABLE %s (%s);" % (
            table_name, ",".join([' '.join(x) for x in zip(columns, column_types)])
        )
        self._conn.execute(create_table)
        self._conn.commit()

    def drop_table(self, table_name):
        """ Create a table with given columns and types

        :param table_name: the table name to drop
        :type table_name: str
        """

        create_table = "DROP TABLE %s;" % table_name
        self._conn.execute(create_table)
        self._conn.commit()

    def return_column_list(self, table_name):
        """

        :param table_name: name of the target table
        :return: list of column names and types
        """

        return_tables = '''SELECT * FROM %s''' % table_name
        result = self._conn.execute(return_tables)
        names = list(map(lambda x: x[0], result.description))
        return names

    def return_table_list(self):
        """ return all the tables in the current db.
        """

        return_tables = 'SELECT name from sqlite_master where type= "table";'
        result = self._conn.execute(return_tables)
        return result.fetchall()

    def return_table_number_row(self, table_name):
        """
        :param table_name: name of the target table
        :return: int
        """
        command = 'SELECT COUNT(1) from %s;' % table_name
        result = self._conn.execute(command)
        return result.fetchall()

    def get_ids(self, table_name):
        """
        This function will get the primary ids of a table
        :param table_name: the table name to retrieve
        :type table_name: str
        :return:
        """
        middle_result = self.get_columns(table_name=table_name, columns=['_id'])
        result = list()
        for tmp_row in middle_result:
            result.append(tmp_row['_id'])
        return set(result)

    def get_columns(self, table_name, columns):
        """ Get column information from a table

        :param table_name: the table name to retrieve
        :type table_name: str
        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """

        select_table = "SELECT %s FROM %s;" % (",".join(columns), table_name)
        result = list(map(lambda x: OrderedDict(zip(columns, x)), self._conn.execute(select_table)))
        return result

    def add_column(self, table_name, column_name, column_type):
        """
        This function adds a new column to an existing table
        :param table_name: the selected table to be added
        :param column_name: the name of added column
        :param column_type: the type of the added column
        :return:
        """

        addColumn = "ALTER TABLE %s ADD COLUMN %s %s" % (table_name, column_name, column_type)
        self._conn.execute(addColumn)
        self._conn.commit()

    def select_row(self, table_name, _id, columns):
        """ Select a row from a table
        (suggestion: consider to use `select_rows` if you want to retrieve multiple rows)

        :param table_name: the table name to retrieve
        :type table_name: str
        :param _id: the row id
        :type _id: str
        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a retrieved row
        :rtype: Dict[str, object]
        """

        select_table = "SELECT %s FROM %s WHERE _id=?;" % (",".join(columns), table_name)
        result = list(self._conn.execute(select_table, [_id]))
        if len(result) == 0:
            return None
        else:
            return OrderedDict(zip(columns, result[0]))

    def select_rows(self, table_name, _ids, columns):
        """ Select rows from a table

        :param table_name: the table name to retrieve
        :type table_name: str
        :param _ids: the row ids
        :type _ids: List[str]
        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: retrieved rows
        :rtype: List[Dict[str, object]]
        """

        if len(_ids) > 0:
            row_cache = dict()
            result = []
            for idx in range(0, len(_ids), self.chunksize):
                select_table = "SELECT %s FROM %s WHERE _id IN ('%s');" % (
                    ",".join(columns), table_name, "','".join(_ids[idx:idx + self.chunksize])
                )
                result.extend(list(self._conn.execute(select_table)))
            for x in result:
                exact_match_row = OrderedDict(zip(columns, x))
                row_cache[exact_match_row["_id"]] = exact_match_row
            exact_match_rows = []
            for _id in _ids:
                exact_match_rows.append(row_cache.get(_id, None))
            return exact_match_rows
        else:
            return []

    def insert_row(self, table_name, row):
        """ Insert a row into a table
        (suggestion: consider to use `insert_rows` if you want to insert multiple rows)

        :param table_name: the table name to insert
        :type table_name: str
        :param row: the row to insert
        :type row: Dict[str, object]
        """

        insert_table = "INSERT INTO %s VALUES (%s)" % (table_name, ",".join(['?'] * (len(row))))
        self._conn.execute(insert_table, list(row.values()))
        self._conn.commit()

    def insert_rows(self, table_name, rows):
        """ Insert several rows into a table

        :param table_name: the table name to insert
        :type table_name: str
        :param rows: the rows to insert
        :type rows: List[Dict[str, object]]
        """

        if len(rows) > 0:
            insert_table = "INSERT INTO %s VALUES (%s)" % (table_name, ",".join(['?'] * (len(next(iter(rows))))))
            try:
                self._conn.executemany(insert_table, [list(row.values()) for row in rows])
                self._conn.commit()
            except sqlite3.InterfaceError:
                print(rows)
                self._conn.executemany(insert_table, [list(row.values()) for row in rows])

    def get_update_op(self, update_columns, operator):
        """ Get an update operator based on columns and a operator

        :param update_columns: a list of columns to update
        :type update_columns: List[str]
        :param operator: an operator that applies to the columns, including "+", "-", "*", "/", "="
        :type operator: str
        :return: an operator that suits the backend database
        :rtype: str
        """

        if operator in "+-*/":
            update_ops = []
            for update_column in update_columns:
                update_ops.append(update_column + "=" + update_column + operator + "?")
            return ",".join(update_ops)
        elif operator == "=":
            update_ops = []
            for update_column in update_columns:
                update_ops.append(update_column + "=?")
            return ",".join(update_ops)
        else:
            raise NotImplementedError

    @staticmethod
    def _update_update_op(row, update_op, update_columns):
        update_op_sp = update_op.split('?')
        while len(update_op_sp) >= 0 and update_op_sp[-1] == '':
            update_op_sp.pop()
        assert len(update_op_sp) == len(update_columns)
        new_update_op = []
        for i in range(len(update_op_sp)):
            new_update_op.append(update_op_sp[i])
            if isinstance(row[update_columns[i]], str):
                new_update_op.append("'" + row[update_columns[i]].replace("'", "''") + "'")
            else:
                new_update_op.append(str(row[update_columns[i]]))
        return ''.join(new_update_op)

    def update_row(self, table_name, row, update_op, update_columns):
        """ Update a row that exists in a table
        (suggestion: consider to use `update_rows` if you want to update multiple rows)

        :param table_name: the table name to update
        :type table_name: str
        :param row: a new row
        :type row: Dict[str, object]
        :param update_op: an operator that returned by `get_update_op`
        :type update_op: str
        :param update_columns: the columns to update
        :type update_columns: List[str]
        """

        update_table = "UPDATE %s SET %s WHERE _id=?" % (table_name, update_op)
        self._conn.execute(update_table, [row[k] for k in update_columns] + [row["_id"]])
        self._conn.commit()

    def update_rows(self, table_name, rows, update_ops, update_columns):
        """ Update rows that exist in a table

        :param table_name: the table name to update
        :type table_name: str
        :param rows: new rows
        :type rows: List[Dict[str, object]]
        :param update_ops: operator(s) that returned by `get_update_op`
        :type update_ops: Union[List[str], str]
        :param update_columns: the columns to update
        :type update_columns: List[str]
        """

        if len(rows) > 0:
            if isinstance(update_ops, (tuple, list)):  # +-*/
                assert len(rows) == len(update_ops)
                # group rows by op to speed up
                update_op_collections = defaultdict(list)  # key: _update_update_op
                for i, row in enumerate(rows):
                    # self.update_row(row, table_name, update_ops[i], update_columns)
                    new_update_op = self._update_update_op(row, update_ops[i], update_columns)
                    update_op_collections[new_update_op].append(row)
                for new_update_op, op_rows in update_op_collections.items():
                    _ids = [row["_id"] for row in op_rows]
                    for idx in range(0, len(_ids), self.chunksize):
                        update_table = "UPDATE %s SET %s WHERE _id IN ('%s');" % (
                            table_name, new_update_op, "','".join(_ids[idx:idx + self.chunksize])
                        )
                        self._conn.execute(update_table)
            else:  # =
                update_op = update_ops
                # group rows by new values to speed up
                value_collections = defaultdict(list)  # key: values of new values
                for row in rows:
                    # self.update_row(row, table_name, update_op, update_columns)
                    value_collections[json.dumps([row[k] for k in update_columns])].append(row)
                for new_update_op, op_rows in value_collections.items():
                    new_update_op = self._update_update_op(op_rows[0], update_op, update_columns)
                    _ids = [row["_id"] for row in op_rows]
                    for idx in range(0, len(_ids), self.chunksize):
                        update_table = "UPDATE %s SET %s WHERE _id IN ('%s');" % (
                            table_name, new_update_op, "','".join(_ids[idx:idx + self.chunksize])
                        )
                        self._conn.execute(update_table)
            self._conn.commit()

    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        """ Retrieve rows by specific keys in some order

        :param table_name: the table name to retrieve
        :type table_name: str
        :param bys: the given columns to match
        :type bys: List[str]
        :param keys: the given values to match
        :type keys: List[str]
        :param columns: the given columns to retrieve
        :type columns: List[str]
        :param order_bys: the columns whose value are used to sort rows
        :type order_bys: List[str]
        :param reverse: whether to sort in a reversed order
        :type reverse: bool
        :param top_n: how many rows to return, default `None` for all rows
        :type top_n: int
        :return: retrieved rows
        :rtype: List[Dict[str, object]]
        """

        key_match_events = []
        select_table = "SELECT %s FROM %s WHERE %s" % (
            ",".join(columns), table_name, " AND ".join(["%s=?" % by for by in bys])
        )
        if order_bys:
            select_table += " ORDER BY %s %s" % (",".join(order_bys), "DESC" if reverse else "ASC")
        if top_n:
            select_table += " LIMIT %d" % top_n
        select_table += ";"
        for x in self._conn.execute(select_table, keys):
            key_match_event = OrderedDict(zip(columns, x))
            key_match_events.append(key_match_event)
        return key_match_events

    def get_rows_by_single_key_multiple_values(self, table_name, by, keys, columns, order_bys=None, reverse=False, top_n=None):
        """ Retrieve rows by specific keys in some order

        :param table_name: the table name to retrieve
        :type table_name: str
        :param by: target column to match
        :type by: str
        :param keys: the given values to match
        :type keys: List[str]
        :param columns: the given columns to retrieve
        :type columns: List[str]
        :param order_bys: the columns whose value are used to sort rows
        :type order_bys: List[str]
        :param reverse: whether to sort in a reversed order
        :type reverse: bool
        :param top_n: how many rows to return, default `None` for all rows
        :type top_n: int
        :return: retrieved rows
        :rtype: List[Dict[str, object]]
        """

        key_match_events = []
        new_keys = list()
        for tmp_key in keys:
            new_keys.append("'"+tmp_key+"'")
        select_table = "SELECT %s FROM %s WHERE %s IN (%s)" % (
            ",".join(columns), table_name, by, ", ".join(new_keys)
        )
        if order_bys:
            select_table += " ORDER BY %s %s" % (",".join(order_bys), "DESC" if reverse else "ASC")
        if top_n:
            select_table += " LIMIT %d" % top_n
        select_table += ";"
        for x in self._conn.execute(select_table):
            key_match_event = OrderedDict(zip(columns, x))
            key_match_events.append(key_match_event)
        return key_match_events

    def create_index(self, table_name, column, index_name):
        """
        This function creates an index for a column
        :param table_name: the table name to create index
        :type table_name: str
        :param column: the column name to create index
        :type column: str
        :param index_name: name of the created index
        :type index_name: str
        :return:
        """
        create_indx = "CREATE INDEX %s ON %s (%s);" % (index_name, table_name, column)
        self._conn.execute(create_indx)
        self._conn.commit()

    def drop_index(self, index_name):
        drop_indx = "DROP INDEX %s;" % index_name
        self._conn.execute(drop_indx)
        self._conn.commit()
