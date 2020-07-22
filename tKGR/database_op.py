import os
import sys
import time

from typing import List
import sqlite3
from sqlite3 import Error
import pymongo

def create_mongo_connection(IP_ADDRESS, DATABASE='tKGR', USER='peng', PASSWORD='siemens'):
    client = pymongo.MongoClient("mongodb://{}:{}@{}/{}".format(USER, PASSWORD, IP_ADDRESS, DATABASE))
    db = getattr(client, DATABASE)
    print("Connection to {}/{} established".format(IP_ADDRESS, DATABASE))
    return db

def register_query_mongo(db, src_idx_l: List[int], rel_idx_l: List[int], cut_time_l: List[int], experiment_info: dict) -> List[int]:
    for src, rel, ts in zip(src_idx_l, rel_idx_l, cut_time_l):
        query = {'subject': int(src), 'relation': int(rel), 'timestamp': int(ts)}
        query.update(experiment_info)
        return db['analysis'].insert_one(query).inserted_id

def insert_a_task_mongo(db, args, checkpoint_dir, git_hash, git_comment, device):
    task = vars(args)
    task['git_hash'] = git_hash
    task['git_comment'] = git_comment
    task['checkpoint_dir'] = checkpoint_dir
    task['aws_device'] = device
    return db['tasks'].insert_one(task).inserted_id

def insert_a_evaluation_mongo(db, checkpoint_dir, epoch, performance):
    log = {'checkpoint_dir': checkpoint_dir, 'epoch':{}}
    performance_key = ['training_loss', 'validation_loss', 'HITS_1_raw', 'HITS_3_raw', 'HITS_10_raw',
                       'HITS_INF', 'MRR_raw', 'HITS_1_fil', 'HITS_3_fil', 'HITS_10_fil', 'MRR_fil']
    performance_dict = {k: float(v) for k, v in zip(performance_key, performance)}
    checkpoint_id = db['logging'].find_one({'checkpoint_dir':checkpoint_dir})
    if checkpoint_id:
        db['logging'].update_one({"_id": checkpoint_id}, {"$set":{"epoch":{epoch: performance_dict}}})
    else:
        log = {'checkpoint_dir': checkpoint_dir, 'epoch': {epoch:performance_dict}}
        db['logging'].insert_one(log)

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

