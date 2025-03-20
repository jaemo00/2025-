from aiohttp import web
import json
import pathlib
import os
from diffusers import DiffusionPipeline


pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipeline.to("cuda")



routes = web.RouteTableDef()
BASE_DIR = pathlib.Path(__file__).parent
IMAGE_DIR = os.path.dirname(os.path.abspath(__file__))


@routes.get('/')
async def serve_html(request):
    return web.FileResponse(BASE_DIR/"index.html")


@routes.get('/image')
async def serve_image(request):
    return web.FileResponse(BASE_DIR/"image_of_squirrel_painting.png")


@routes.post("/submit")
async def handle_post(request):
    """클라이언트에서 JSON 데이터를 받아 응답하는 핸들러"""
    try:


        data = await request.json()
        text = data.get("text", "")
        image = pipeline(text).images[0]

        image_filename=text+'.png'
        image.save(image_filename)
        
        print(f"Received text: {image_filename}") 

        image_path = os.path.join(IMAGE_DIR, image_filename)
       
        if not os.path.exists(image_path):
            image_path = os.path.join(IMAGE_DIR, "default.png")  # 기본 이미지 반환
        print(f"path:{image_path}")
        return web.json_response({"status": "success", "image_url": f"/{image_filename}"})
        

    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=400)
    
@routes.get("/{filename}")
async def serve_image(request):
    """이미지 제공 엔드포인트"""
    filename = request.match_info["filename"]
    image_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(image_path):  # 파일이 없으면 기본 이미지 반환
        image_path = os.path.join(IMAGE_DIR, "default.png")

    return web.FileResponse(image_path)


app = web.Application()
app.add_routes(routes)

if __name__=="__main__":
    web.run_app(app, host='127.0.0.1', port=8080)