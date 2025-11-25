import cv2
import requests
import numpy as np

url = "http://172.20.10.3:81/stream"

r = requests.get(url, stream=True)
bytes_buffer = b""

for chunk in r.iter_content(chunk_size=4096):
    bytes_buffer += chunk

    # Look for the frame boundary header
    boundary = bytes_buffer.find(b'\r\n\r\n')
    if boundary == -1:
        continue

    # After the header comes the JPEG bytes
    jpg_start = bytes_buffer.find(b'\xff\xd8', boundary)
    jpg_end = bytes_buffer.find(b'\xff\xd9', jpg_start)

    if jpg_start != -1 and jpg_end != -1:
        jpg = bytes_buffer[jpg_start:jpg_end + 2]
        bytes_buffer = bytes_buffer[jpg_end + 2:]

        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("ESP32 Stream", img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
