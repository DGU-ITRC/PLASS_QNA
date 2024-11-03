import model
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="template")

SERVER_INFO = {
    "name": "Question and Answer API Server",
    "version": "1.0.0",
    "port": 50001
}

@app.get("/")
async def demo(request: Request):
    """
    데모를 실행할 수 있는 페이지를 반환합니다.

    Html file: /templates/index.html

    Args:
        request (Request): FastAPI Request 객체

    Returns:
        TemplateResponse: HTML 파일
    """
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/predict")
async def predict(context: str = Form(...), question: str = Form(...)):
    """
    질문에 대한 답변을 반환합니다.

    Args:
        context (str): 질문에 대한 문맥
        question (str): 질문
    
    Returns:
        dict: 질문에 대한 답변
    """
    response = model.predict(context, question)
    json_response = jsonable_encoder(response)
    return JSONResponse(content=json_response)