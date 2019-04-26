import requests

url = "http://104.215.146.190:8080/predict"
files = {'file': open('storage/Record.mp3', 'rb')}
r = requests.post(url, files=files)
response = r.text
print(r, response)