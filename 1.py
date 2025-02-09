from pymongo import MongoClient

# Connect to MongoDB
db_client = MongoClient("mongodb://localhost:27017/")

# Access traffic database
db = db_client["traffic"]

print(db.list_collection_names())

# Access traffic collection
traffic_col = db["signals"]

# print(signal_times["lane_1"]["green"])
print(traffic_col.find_one({"state": "Maharashtra"}))
