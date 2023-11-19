# PD-Fast-Yolo
This project is a plate detection for iranian number plates and uses yolo 7 with fastapi and uvicorn as a webserver. 

# results
![alt text](https://github.com/AshkanVakili9/PD-Fast-Yolo/blob/main/results/1.jpg)
![alt text](https://github.com/AshkanVakili9/PD-Fast-Yolo/blob/main/results/2.jpg)
![alt text](https://github.com/AshkanVakili9/PD-Fast-Yolo/blob/main/results/3.jpg)
![alt text](https://github.com/AshkanVakili9/PD-Fast-Yolo/blob/main/results/4.jpg)
![alt text](https://github.com/AshkanVakili9/PD-Fast-Yolo/blob/main/results/5.jpg)

# Detail About The Project
plate detection works pretty good with most of the car pics but I used EasyOCR for the read the plates and it's not good but im working on a persian ocr model. 
the project idea all comes from "https://www.youtube.com/watch?v=bgAUHS1Adzo" Maryam Sadeghi        

# Get Start With Project
1. After Clone the project create a virtual env with this command "python -m venv venv" 
2.  activate it by this command ".\venv\Scripts\activate"
3.  go to the yolo7 folder "cd yolo7"
4.  install the packages "pip install -r requirements.txt"
5.  run the project "uvicorn main:app --reload"
6.  open the browser and go to this URL "http://127.0.0.1:8000/docs"
