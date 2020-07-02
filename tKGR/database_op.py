import os

from typing import List
import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def create_task_table(conn, hyperparameters: List[str], args, table_name = "tasks"):
    """

    :param conn:
    :param hyperparameters: hyparameters to be stored in database, except checkpoint_dir, which is primary key
    :param args:
    :return:
    """
    with conn:
        cur = conn.cursor()
        table_exists = False
        # get the count of tables with the name
        cur.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{}' '''.format(table_name))

        # if the count is 1, then table exists
        if cur.fetchone()[0] == 1:
            print("table exists")
            table_exists = True
        if table_exists:
            columns = [i[1] for i in cur.execute('PRAGMA table_info({})'.format(table_name))]
        create_table_sql = "CREATE TABLE IF NOT EXISTS {} (checkpoint_dir TEXT PRIMARY KEY, ".format(table_name)
        for hp in hyperparameters:
            try:
                arg = getattr(args, hp)
                if isinstance(arg, int):
                    arg_type = "INTEGER"
                elif isinstance(arg, float):
                    arg_type = "REAL"
                elif isinstance(arg, bool):
                    arg_type = "INTEGER"
                    arg = int(arg)
                elif isinstance(arg, str):
                    arg_type = "TEXT"
                else:
                    raise AttributeError("Doesn't support this data type in create_task_table, daatabase_op.py")
            except:
                raise AttributeError("'Namespace' object has no attribute "+hp)
            if table_exists:
                if hp not in columns:
                    cur.execute('ALTER TABLE {} ADD COLUMN {} {}'.format(table_name, hp, arg_type))
            create_table_sql += " ".join([hp, arg_type])+", "

        if table_exists:
            if "git_hash" not in columns:
                cur.execute('ALTER TABLE {} ADD COLUMN git_hash TEXT'.format(table_name))
        create_table_sql += "git_hash TEXT NOT NULL);"
        if not table_exists:
            create_table(conn, create_table_sql)

def insert_into_task_table(conn, hyperparameters, args, checkpoint_dir, git_hash, table_name='tasks'):
    with conn:
        cur = conn.cursor()
        # get the count of tables with the name
        cur.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{}' '''.format(table_name))

        # if the count is 1, then table exists
        if cur.fetchone()[0] != 1:
            raise Error("table doesn't exist")

        placeholders = ', '.join('?' * (len(hyperparameters)+2))
        sql_hp = 'INSERT OR IGNORE INTO {}({}) VALUES ({})'.format(table_name, 'checkpoint_dir, git_hash, '+', '.join(hyperparameters), placeholders)
        sql_hp_val = [checkpoint_dir, git_hash]
        for hp in hyperparameters:
            try:
                arg = getattr(args, hp)
                if isinstance(arg, bool):
                    arg = int(arg)
                sql_hp_val.append(arg)

            except:
                raise AttributeError("'Namespace' object has no attribute "+hp)
        cur.execute(sql_hp, sql_hp_val)
        task_id = cur.lastrowid
        return task_id





def create_logging_table(conn, table_name = "logging"):
    """

    :param conn:
    :param hyperparameters: hyparameters to be stored in database, except checkpoint_dir, which is primary key
    :param args:
    :return:
    """
    with conn:
        logging_col = (
            'checkpoint_dir', 'epoch', 'training_loss', 'validation_loss', 'HITS_1_raw', 'HITS_3_raw', 'HITS_10_raw',
            'HITS_INF', 'MRR_raw', 'HITS_1_fil', 'HITS_3_fil', 'HITS_10_fil', 'MRR_fil')
        sql_create_loggings_table = """ CREATE TABLE IF NOT EXISTS logging (
        checkpoint_dir text NOT NULL,
        epoch integer NOT NULL,
        training_loss real,
        validation_loss real,
        HITS_1_raw real,
        HITS_3_raw real,
        HITS_10_raw real,
        HITS_INF real,
        MRR_raw real,
        HITS_1_fil real,
        HITS_3_fil real,
        HITS_10_fil real,
        MRR_fil real,
        PRIMARY KEY (checkpoint_dir, epoch),
        FOREIGN KEY (checkpoint_dir) REFERENCES tasks (checkpoint_dir)
        );"""
        create_table(conn, sql_create_loggings_table)


def insert_into_logging_table(conn, checkpoint_dir, epoch, performance, table_name='logging'):
    with conn:
        cur = conn.cursor()
        # get the count of tables with the name
        cur.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{}' '''.format(table_name))

        # if the count is 1, then table exists
        if cur.fetchone()[0] != 1:
            raise Error("table doesn't exist")

        logging_col = (
            'checkpoint_dir', 'epoch', 'training_loss', 'validation_loss', 'HITS_1_raw', 'HITS_3_raw', 'HITS_10_raw',
            'HITS_INF', 'MRR_raw', 'HITS_1_fil', 'HITS_3_fil', 'HITS_10_fil', 'MRR_fil')

        placeholders = ', '.join('?' * len(logging_col))
        sql_logging = 'INSERT OR IGNORE INTO {}({}) VALUES ({})'.format(table_name, ', '.join(logging_col), placeholders)
        sql_logging_val = [checkpoint_dir, epoch] + performance
        cur.execute(sql_logging, sql_logging_val)

