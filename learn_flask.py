from flask import Flask, render_template, request
import time

app = Flask(__name__)

from flask_uploads import UploadSet, configure_uploads, IMAGES
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static'
configure_uploads(app, photos)

@app.route("/")
def index():
    return render_template('home.html')
 
@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')

@app.route("/upload" )
def upload():
    return render_template('upload.html')

@app.route("/save_img", methods=['GET', 'POST'])
def save():
    print(request.files)
    print(request.method)
    
    if request.method == 'POST' and 'fileField' in request.files:
        filename = photos.save(request.files['fileField'])
    return render_template('display.html', filename=filename)
 
if __name__ == "__main__":
    app.run(debug=True)