from flask import Flask, render_template, Response
from camera import VideoCamera
from person_counter import PersonCounter

app = Flask(__name__)

@app.route('/')
def index():
    # render page
    return render_template('index.html')


@app.route('/person')
def person():
    # render page
    return render_template('person.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (
                b'--frame\r\n' +
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame +
                b'\r\n\r\n'
               )


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/person_feed')
def person_feed():
    return Response(gen(PersonCounter(input_video='video/example_01.mp4')), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)