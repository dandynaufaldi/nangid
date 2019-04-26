import os
import requests
import sys
import time
"""
- tiap 30 detik, record 9 detik pake system call ke arecord
- request post ke server
- olah response
"""
FILENAME = "recording.wav"
URL = "http://104.215.146.190:8080/predict"
# URL = "http://0.0.0.0:8080/predict"


def get_prediction():
    files = {'file': open(FILENAME, 'rb')}
    response = requests.post(URL, files=files)
    return response.text


if __name__ == "__main__":
    while True:
        try:
            # 'arecord --device="hw:1,0" -d 5 -f S16_LE -c 2 -r 44100 -t wav hehe.wav'
            print('[RECORD]')
            os.system('arecord --device="hw:1,0" -d 9 -f S16_LE -c 2 -r 44100 -t wav {}'.format(FILENAME))
            # os.system("arecord -D plughw:1,0 -d 9 -f S16_LE -c1 -r22050 -t wav {}".format(FILENAME))
            # print(os.listdir("."))
            prediction = get_prediction()
            print(prediction)
            time.sleep(30)
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            pass
