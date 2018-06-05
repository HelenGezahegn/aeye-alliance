from flask import Flask
from flask import request 
app = Flask(__name__)

# Here, we will import the model file
# import file 

@app.route('/submit/<filename>')
def submit_image(filename):

    # Reshape the image for the model 
    print(filename)
    return "Image submitted" 

if __name__ == "__main__":
    app.run()
