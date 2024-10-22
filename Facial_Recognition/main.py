from pydantic import BaseModel
from typing import List
import face_recognition
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv('MONGO_DB_URL'))
db = client[os.getenv('DATABASE_NAME')]  
users_collection = db[os.getenv('COLLECTION_NAME')]

class UserSchema(BaseModel):
    userId: str
    profileEncodings: List[List[float]] 

async def load_image_file(file):
    return face_recognition.load_image_file(file)

async def face_encodings(image):
    return face_recognition.face_encodings(image)

async def compare_faces(profile_encoding, unknown_encoding):
    return face_recognition.compare_faces([profile_encoding], unknown_encoding)

async def insert_profile_encoding(userId, encoding):
    return users_collection.insert_one({"userId": userId, "profileEncodings": encoding})

async def find_user(userId):
    return users_collection.find_one({"userId": userId})

async def update_profile_encoding(userId, encoding):
    return users_collection.update_one(
        {"userId": userId},
        {"$set": {"profileEncodings": encoding}}
    )
