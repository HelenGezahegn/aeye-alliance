from flask import Flask, render_template
app = Flask(__name__)
 
@app.route("/")
def index():
    return render_template('home.html')
 
@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')

# @app.route("/hello")
# def hello():
#     return "Hello World!"
 
# @app.route("/members")
# def members():
#     return "Members"
 
# @app.route("/members/<string:name>/")
# def getMember(name):
#     return name
 
if __name__ == "__main__":
    app.run(debug=True)