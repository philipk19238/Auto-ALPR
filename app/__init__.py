from flask import Flask


UPLOAD_FOLDER = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/app/static/videos'
IMAGE_FOLDER = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/app/static/images'
ALLOWED_EXTENSIONS = ['mp4', 'avi']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config["SECRET_KEY"] = "OCML3BRawWEUeaxcuKHLpw"

from app import routes
