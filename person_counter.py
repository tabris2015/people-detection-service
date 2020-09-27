import cv2
import numpy as np
import imutils
import time
import dlib
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from imutils.video import VideoStream, FPS

# ancho de la imagen
IM_WIDTH = 500





TIME_LIMIT = 200

# clases de la red neuronal
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


class PersonCounter(object):
    # ancho de la imagen
    IM_WIDTH = 500
    N_TRACE = 60
    ## sectores de la tienda
    sectors = {  # sector		start(x, y)					end(x, y)
        'pasillo': ((IM_WIDTH // 3, 0), (IM_WIDTH * 2 // 3, 300)),  #
        'tienda1': ((0, 0), (IM_WIDTH // 3, 100)),
        'tienda2': ((0, 100), (IM_WIDTH // 3, 200)),
        'tienda3': ((0, 200), (IM_WIDTH // 3, 300)),
        'tienda4': ((IM_WIDTH * 2 // 3, 0), (IM_WIDTH, 100)),
        'tienda5': ((IM_WIDTH * 2 // 3, 100), (IM_WIDTH, 200)),
        'tienda6': ((IM_WIDTH * 2 // 3, 200), (IM_WIDTH, 300))
    }
    # puntos
    sector_points = {  # sector	[initial_points, counter]
        'pasillo': [100, 0],
        'tienda1': [100, 0],
        'tienda2': [100, 0],
        'tienda3': [100, 0],
        'tienda4': [100, 0],
        'tienda5': [100, 0],
        'tienda6': [100, 0]
    }
    def __init__(self, input_video=None, skip_frames=30, model_dir='model', model_name='MobileNetSSD_deploy', confidence=0.4):
        # dimensiones del frame
        self.W = None
        self.H = None
        self.skip_frames = skip_frames
        self.confidence = confidence
        self.input_video = input_video
        self.model_dir = model_dir
        self.model_name = model_name
        self.status = 'Init'
        self.trackers = []
        self.last_frame = np.zeros(10)
        # variables para contar personas
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
        if not self.input_video:
            self.vs = cv2.VideoCapture(0)
            # self.vs = VideoStream(src=0).start()
        else:
            self.vs = cv2.VideoCapture(input_video)

        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackableObjects = {}
        # estimador de fps
        self.fps = FPS().start()
        self.net = cv2.dnn.readNetFromCaffe(
            self.model_dir + '/' + self.model_name + '.prototxt',
            self.model_dir + '/' + self.model_name + '.caffemodel'
        )

    def in_rect(self, position, start, end):
        return position[0] >= start[0] and position[0] <= end[0] and position[1] >= start[1] and position[1] <= end[1]

    def __del__(self):
        self.video.release()
        self.fps.stop()

    def get_sector(self, position, sectors):
        for sector, (start, end) in sectors.items():
            if self.in_rect(position, start, end):
                return sector

    def get_frame(self):
        time.sleep(0.1)
        """Se ejecuta en cada frame"""
        ret, frame = self.vs.read()
        # frame = frame[1] if not self.input_video else frame
        if not isinstance(frame, (np.ndarray, np.generic)):
            time.sleep(0.2)
            ret, jpeg = cv2.imencode('.jpg', self.last_frame)
            return jpeg.tobytes()

        frame = imutils.resize(frame, width=IM_WIDTH)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        self.status = "Waiting"
        rects = []

        if self.totalFrames % self.skip_frames == 0:
            self.status = "Detecting"
            self.trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            # para cada deteccion
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    self.trackers.append(tracker)
            for objectID, centroid in self.ct.objects.items():
                to = self.trackableObjects.get(objectID, None)
                for cent in to.centroids:
                    sector = self.get_sector(cent, self.sectors)
                    if sector:
                        self.sector_points[sector][1] += 1
                        if self.sector_points[sector][1] > TIME_LIMIT:
                            self.sector_points[sector][0] += 2
                            self.sector_points[sector][1] = 0
                            for sec, points in self.sector_points.items():
                                points[0] -= 1

        else:
            for tracker in self.trackers:
                self.status = 'Tracking'
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        objects = self.ct.update(rects)

        for objectID, centroid in objects.items():
            to = self.trackableObjects.get(objectID, None)
            if not to:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < self.H // 2:
                        self.totalUp += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > self.H // 2:
                        self.totalDown += 1
                        to.counted = True

            self.trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(
                frame,
                text,
                (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            sector = self.get_sector(centroid, self.sectors)
            cv2.putText(
                frame,
                sector,
                (centroid[0] - 10, centroid[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            plot_centroids = to.centroids if len(to.centroids) < self.N_TRACE else to.centroids[-self.N_TRACE:]
            for i in range(0, len(plot_centroids), 4):
                cv2.circle(
                    frame,
                    (plot_centroids[i][0], plot_centroids[i][1]),
                    4,
                    (0, 255, 0),
                    -1
                )

        # for sector, (start, end) in self.sectors.items():
        #     cv2.rectangle(
        #         frame,
        #         start,
        #         end,
        #         (0, 255, 255),
        #         2
        #     )
        #     cv2.putText(
        #         frame,
        #         str(self.sector_points[sector][0]),
        #         (start[0], start[1] + 15),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2
        #     )
        self.totalFrames += 1
        self.fps.update()

        ret, jpeg = cv2.imencode('.jpg', frame)
        self.last_frame = np.copy(frame)
        return jpeg.tobytes()
