from fastapi import FastAPI, UploadFile, File
from src.ai_service import ai_agent

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI"}

@app.post("/inference")
async def inference(img1: UploadFile = File()):
    try:

        res = ai_agent(img1.file.read())
        return {"response" : "200", "msg" : res}
    except Exception as e:
        return {"response": "500","msg" : str(e)}