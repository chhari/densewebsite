from flask import Flask,render_template, request, send_file
import cv2
import infer_website as webinfer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success-table', methods=['POST'])
def success_table():
    global filename
    if request.method=="POST":
        file=request.files['file']
        try:
            myImage = cv2.imread(str(file))
            print("done mama")
            webinfer.infer_method(myImage)
            return render_template("index.html", text="read the file")
        except Exception as e:
            return render_template("index.html", text=str(e))




if __name__=="__main__":
    app.run(debug=True)
