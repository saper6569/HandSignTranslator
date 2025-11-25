import cv2
import urllib.request
import numpy as np

url = "http://172.20.10.3:81/stream"

stream = urllib.request.urlopen(url)
bytes_buffer = b""

while True:
    bytes_buffer += stream.read(1024)

    a = bytes_buffer.find(b'\xff\xd8')   # JPEG start
    b = bytes_buffer.find(b'\xff\xd9')   # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_buffer[a:b+2]
        bytes_buffer = bytes_buffer[b+2:]   # remove processed part

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        cv2.imshow("ESP32 Fast Stream", frame)

        # Low-latency window refresh
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
