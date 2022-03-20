from pymongo import MongoClient


class Mongo:
    def __init__(self):
        self._db = MongoClient('192.168.108.101').thesis

        """self._db.d1.drop()
        self._db.d2.drop()

        self._db.d1.create_index([
            ('first', 1),
            ('second', 1)
        ], unique=True)

        self._db.d1.create_index([
            ('index', 1)
        ], unique=True)

        self._db.d2.create_index([
            ('first', 1),
            ('second', 1)
        ], unique=True)

        self._db.d2.create_index([
            ('index', 1)
        ], unique=True)"""

    def d1_insert_many(self, insert_list):
        self._db.d1.insert_many(insert_list)

    def d2_insert_many(self, insert_list):
        self._db.d2.insert_many(insert_list)

    def d1_get(self):
        return self._db.d1.find({})

    def d2_get(self):
        return self._db.d2.find({})

    def d1_get_by_index_list(self, index_list):
        return self._db.d1.find({'index': {'$in': index_list}})

    def d2_get_by_index_list(self, index_list):
        return self._db.d2.find({'index': {'$in': index_list}})

    def d1_get_count(self):
        return self._db.d1.count_documents({})

    def d2_get_count(self):
        return self._db.d2.count_documents({})

    def d1_get_index_range(self, index_from, index_to):
        return self._db.d1.find({'index': {'$gte': index_from, '$lt': index_to}})

    def d2_get_index_range(self, index_from, index_to):
        return self._db.d2.find({'index': {'$gte': index_from, '$lt': index_to}})

    def d1_bulk_write_update_many(self, update_list):
        self._db.d1.bulk_write(update_list)

    def d2_bulk_write_update_many(self, update_list):
        self._db.d2.bulk_write(update_list)
