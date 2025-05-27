import requests

url = 'http://localhost:5000/api/analyze_video'
video_path = '/home/aico/Documents/GitHub/pilzriskdetection/parmak sıkışma (2).mp4'

with open(video_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print('Status:', response.status_code)
print('Response:', response.json())