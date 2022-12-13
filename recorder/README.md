# Recorder Example
Code in this folder shall demonstrate how to authenticate against a VW² instance, grab stream from a camera and save frames into a file.

## Usage

```
pip install -r requirements.txt
```

```
export KEYCLOAK_CLIENT_SECRET=YOUR_SECRET
```
```
python recorder.py
```


## How things work
TODO
- get KeyCloak token
- get list of cameras
- present list and collect choice
- setup grpc stub
- connect to CameraService and grab stream
- write to file