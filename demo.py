import requests

# The path to your JSON file
json_file_path = 'request_body.json'

# The URL of your FastAPI endpoint
url = "http://127.0.0.1:8000/forecast/"

with open(json_file_path, 'rb') as file:
    files = {'file': (json_file_path, file)}
    # Send a POST request with the file
    response = requests.post(url, files=files)


data = response.text
print(data)
