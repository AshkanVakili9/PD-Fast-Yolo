from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from starlette.requests import Request
from ANPR_ir import get_plates_from_image
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2


app = FastAPI()


app.mount("/ANPR", StaticFiles(directory="ANPR"), name="ANPR")
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, './ANPR/upload/')
savepath = os.path.join(BASE_PATH, './ANPR/results/')


@app.post("/upload-image/")
async def create_upload_file(request: Request, image_name: UploadFile = File(...)):
    filename = image_name.filename
    path_save = os.path.join(UPLOAD_PATH, filename)

    with open(path_save, "wb") as image_file:
        shutil.copyfileobj(image_name.file, image_file)

    plate_image = cv2.imread(path_save)
    # Get the detected image with bounding boxes and labels
    detected_image, plate_text = get_plates_from_image(plate_image)


    # Convert the detected image to base64 format for inclusion in the response
    # _, img_buffer = cv2.imencode(".png", detected_image)
    # img_base64 = base64.b64encode(img_buffer).decode("utf-8")
    
    cv2.imwrite(os.path.join(savepath, filename), detected_image)

    # Return the result as a JSON response with text and the detected image
    response_data = {"text": plate_text, "image_url": filename}
    return JSONResponse(content=response_data)


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(savepath, filename)
    return FileResponse(file_path)
