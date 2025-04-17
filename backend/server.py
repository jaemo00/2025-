import pathlib
import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline

'''
pipe = StableDiffusionPipeline.from_single_file(
    "C:/Users/AhnLab/Desktop/DreamShaper_8.safetensors",  # DreamShaper 파일 경로
    torch_dtype=torch.float16,
    safety_checker=None,  # 필요 시 꺼줄 수 있음
)
pipe = pipe.to("cuda")

generator = torch.manual_seed(33)
'''
class SubmitRequest(BaseModel):
    text: str
#pipe.load_lora_weights("C:/Users/AhnLab/Desktop/sd1.5.safetensors",weight_name="default", lora_scale=0.7) #0.5~1



app = FastAPI()
#BASE_DIR = pathlib.Path(__file__).parent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")  # frontend 폴더 위치

# frontend 폴더 mount
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# static 파일들 mount
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get('/',response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.post("/submit")
async def handle_post(data: SubmitRequest):

    """클라이언트에서 JSON 데이터를 받아 응답하는 핸들러"""
    try:
        text = data.text
        #image = pipe(text, generator=generator, num_inference_steps=30).images[0]

        image_filename=text+'.png'
        image_path = os.path.abspath(os.path.join(STATIC_DIR, image_filename))
        #image.save(image_path)
        
        print(f"Received text: {image_filename}") 

        if not os.path.exists(image_path):
            image_path = os.path.join(STATIC_DIR, "default.png")  # 기본 이미지 반환
        print(f"path:{image_path}")
        return JSONResponse(content={"image_url": image_filename,"status":"success"})
        

    except Exception as e:
        return JSONResponse(content={"image_url": image_filename,"status":"success"})
    



if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.2", port=8000, reload=True)
