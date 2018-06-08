from flask import Flask, render_template, request
import time, CNN, os
import prediction

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

@app.route("/upload")
def upload():
    return render_template('upload.html')

@app.route("/upload_results", methods=['GET', 'POST'])
def save():
    print(request.files)
    print(request.method)
    
    # Save the image in the path
    if request.method == 'POST' and 'fileField' in request.files:
        filename = photos.save(request.files['fileField'])
    
    # print(os.getcwd())
    img_path = "static/" + filename
    predicted_letter = prediction.make_prediction(img_path)
    # time.sleep(1)
    return render_template('display.html', filename=filename, letter=predicted_letter)
    # return render_template('display.html', filename=filename)

if __name__ == "__main__":
    app.run(debug=True)