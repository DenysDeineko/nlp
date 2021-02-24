import re
import json

import nltk
import mysql.connector

from mysql.connector import Error
from environs import Env
from nltk.chunk import RegexpParser, conlltags2tree
from sner import Ner

from keywords import keywords


class Helper:
    @staticmethod
    def get_sentences(raw_string):
        return nltk.sent_tokenize(raw_string.replace('\n', ' '))

    @staticmethod
    def get_tokenized_sentences(sentences):
        return [nltk.word_tokenize(sentence) for sentence in sentences]

    @staticmethod
    def get_parsed_sentences(grammar, trees):
        tag_parser = RegexpParser(grammar)
        return tag_parser.parse_sents(trees)

    @staticmethod
    def extract_entities(parsed_sentences, row):
        extracted_text = []
        entities = []

        for sentence in parsed_sentences:
            extracted_text.extend(extract_text(sentence))
        pattern = re.compile(r"(?:%s)\D*" % "|".join(titles))

        for item in extracted_text:
            if isinstance(item, dict):
                title = ''.join(pattern.findall(item['title']))
                if len(title):
                    first_name, last_name = Helper.get_processed_name(item['name'])
                    if len(first_name) and len(last_name):
                        entities.append({
                            "first_name": first_name,
                            "last_name": last_name,
                            "job_title": Helper.get_processed_title(title),
                            "email": row['email'],
                            "url": row['url']
                        })
        return entities

    @staticmethod
    def get_processed_name(raw_name):
        skip_words = ['emeritus', '\d*|-\d+']
        name = re.sub(r'%s' % '|'.join(skip_words), '', raw_name, flags=re.IGNORECASE).strip()
        split_by_word = name.split()
        if len(split_by_word) > 2:
            name = ' '.join(set(split_by_word))
        first_name, *last_name = name.split()
        if len(last_name) > 5:
            last_name = last_name[:5]
        return first_name, ' '.join(last_name)

    @staticmethod
    def get_processed_title(raw_title):
        skip_words = ['ms.', 'mr.', 'mrs.', 'dr.', '-lrb-', '-rrb-']
        title = re.sub(r'%s' % '|'.join(skip_words), '', raw_title, flags=re.IGNORECASE).strip()
        if len(title) > 255:
            title = title[:255]
        return title

    @staticmethod
    def transform_stanford_name_entity_to_bio(tagged_sent):
        bio_tagged_sent = []
        prev_tag = "O"
        for token, tag in tagged_sent:
            if tag == "O":  # O
                bio_tagged_sent.append((token, tag))
                prev_tag = tag
                continue
            if tag != "O" and prev_tag == "O":  # Begin NE
                bio_tagged_sent.append((token, "B-" + tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag == tag:  # Inside NE
                bio_tagged_sent.append((token, "I-" + tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag != tag:  # Adjacent NE
                bio_tagged_sent.append((token, "B-" + tag))
                prev_tag = tag

        return bio_tagged_sent

    @staticmethod
    def transform_stanford_name_entity_to_tree(ne_tagged_sent):
        ne_tree = []
        if ne_tagged_sent:
            bio_tagged_sent = Helper.transform_stanford_name_entity_to_bio(ne_tagged_sent)
            sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
            sent_pos_tags = [pos for token, pos in nltk.pos_tag(sent_tokens)]
            sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
            ne_tree = conlltags2tree(sent_conlltags)
        return ne_tree

    @staticmethod
    def get_grammar(model):
        grammar = {
            "stanford_ner": r"""NAMED-ENTITY: {<NNP><IN|CC|DT|NN|NNP>*<ORGANIZATION>?<PERSON>}
                                              {<PERSON><NNP><IN|CC|DT|NN|NNP>*<ORGANIZATION>?}""",
            "nltk": r"""NAMED-ENTITY: {<PERSON><NNP><IN|CC|DT|NN|NNP>*}
                                      {<NNP><IN|CC|DT|NN|NNP|>*<PERSON>}
                        PERSON: {<NNP>{2}}"""

        }
        return grammar[model]


def get_db_config():
    env = Env()
    env.read_env()
    return {
        "remote": {
            "host": env("DB_HOST_1"),
            "user": env("DB_USER_1"),
            "password": env("DB_PASSWORD_1"),
            "database": env("DB_USER_1")},
        "local": {
            "host": env("DB_HOST_2"),
            "user": env("DB_USER_2"),
            "password": env("DB_PASSWORD_2"),
            "database": env("DB_USER_2")
        }
    }


def connect(host, user, password, database):
    connection = None
    try:
        connection = mysql.connector.connect(host=host, user=user, passwd=password, database=database)
        cursor = connection.cursor()
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection, cursor


def get_data(config):
    connection, cursor = connect(**config)
    cursor.execute("""select id, url, email, json from data""")
    data = [{'id': row[0], 'url': row[1], 'email': row[2], 'supposed_string': "".join(json.loads(row[3]).values())}
            for row in cursor.fetchall()]
    cursor.close()
    return data


def use_stanford_ner(data):
    entities = []
    helper = Helper
    tagger = Ner(host='localhost', port=9199)

    for row in data:
        sentences = helper.get_sentences(row['supposed_string'])
        ne_sentences = [tagger.get_entities(sentence) for sentence in sentences]
        tagged_sentences = [helper.transform_stanford_name_entity_to_tree(sentence) for sentence in ne_sentences]
        grammar = helper.get_grammar('stanford_ner')
        parsed_sentences = helper.get_parsed_sentences(grammar, tagged_sentences)
        entities.extend(helper.extract_entities(parsed_sentences, row))
    return entities


def use_nltk(data):
    entities = []
    helper = Helper

    for row in data:
        sentences = helper.get_sentences(row['supposed_string'])
        tokenized_sentences = helper.get_tokenized_sentences(sentences)
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        grammar = helper.get_grammar('nltk')
        parsed_sentences = helper.get_parsed_sentences(grammar, tagged_sentences)
        entities.extend(helper.extract_entities(parsed_sentences, row))
    return entities


def extract_text(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NAMED-ENTITY':
            person = t.pop(t.index(next(t.subtrees(filter=lambda x: x.label() == "PERSON"))))
            if len(person) > 1:
                entity_names.append({
                    "title": ' '.join([child[0] for child in t.flatten()]),
                    "name": ' '.join(extract_text(person))
                })
        elif t.label() == 'PERSON':
            return [child[0] for child in t]
        else:
            for child in t:
                entity_names.extend(extract_text(child))

    return entity_names


def save_results(config, data):
    connection, cursor = connect(**config)
    cursor.execute("""CREATE TABLE IF NOT EXISTS named_entities(
                        id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        first_name VARCHAR(50) NOT NULL,
                        last_name VARCHAR(70) NOT NULL,
                        job_title VARCHAR(255) NOT NULL,
                        email VARCHAR(100),
                        url VARCHAR(255) NOT NULL)""")
    prepared_data = [tuple(item.values()) for item in data]
    cursor.executemany(("INSERT INTO named_entities (first_name, last_name, job_title, email, url) "
                        "VALUES (%s, %s, %s, %s, %s)"), prepared_data)
    connection.commit()
    row_count = cursor.rowcount
    cursor.close()
    return row_count


if __name__ == "__main__":
    titles = keywords().get('academic_title', [])
    config = get_db_config()
    data = get_data(config['remote'])
    entities = use_stanford_ner(data)
    save_results(config['local'], entities)
