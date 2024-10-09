from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from Facial_Recognition.main import (
    load_image_file,
    face_encodings,
    insert_profile_encoding,
    update_profile_encoding,
    find_user,
    compare_faces,
)
import numpy as np

app = FastAPI()

@app.post("/store_encoding")
async def store_encoding_api(userId: str = Form(...), file: UploadFile = File(...)):
    image = await load_image_file(file=file.file)
    face_encoding = await face_encodings(image)

    if len(face_encoding) == 0:
        return {"error": "No face detected in the image"}
    
    encoding = face_encoding[0].tolist()
    await insert_profile_encoding(userId, encoding)
    
    return {"message": "Profile encoding stored successfully!"}

@app.put("/update_encoding")
async def update_profile_encoding_api(userId: str = Form(...), file: UploadFile = File(...)):
    image = await load_image_file(file=file.file)
    face_encoding = await face_encodings(image)

    if not face_encoding:
        raise HTTPException(status_code=400, detail="No face found in the image.")
    
    encodings = face_encoding[0].tolist()
    result = await update_profile_encoding(userId=userId, encoding=encodings)

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "Profile encoding updated successfully!"}

@app.post("/recognize")
async def recognize_user_api(userId: str = Form(...), file: UploadFile = File(...)):
    user = await find_user(userId=userId)

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    unknown_image = await load_image_file(file=file.file)
    unknown_encodings = await face_encodings(image=unknown_image)

    if not unknown_encodings:
        raise HTTPException(status_code=400, detail="No face found in the image.")

    unknown_encoding = unknown_encodings[0]
    profile_encoding = np.array(user["profileEncodings"])
    results = await compare_faces(profile_encoding=profile_encoding, unknown_encoding=unknown_encoding)

    if results[0]:
        return {"message": "User matched!"}
    
    return {"message": "user mismatch!"}

@app.get("/")
async def hello():
    return {"message": "Your server is live."}
