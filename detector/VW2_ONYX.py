import os
import cv2
import grpc
import json
import time
import argparse
import requests
import traceback
import numpy as np
from getpass import getpass
from typing import Callable, List
from datetime import datetime, timezone
from keycloak.keycloak_openid import KeycloakOpenID 
from icv.camera.v1 import CameraController_pb2_grpc
from icv.camera.v1.CameraController_pb2 import GetImageRequest, StreamRequest

# YOLOv7 code - move YOLOv7 and models folder from ibaiGorordo's ONNX-YOLOv7-Objet-Detection git folder (https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection)
from YOLOv7 import YOLOv7

CLIENT_ID = "ID_HERE"
REALM = "icv"

BASE_PLATFORM_URL="https://carmel2.vw2-demospace.de/carmel-smart-cities-"
CAMERASERVICE_URL=f"{BASE_PLATFORM_URL}cameraservice"
AUTH_URL=f"{BASE_PLATFORM_URL}auth/"

def prompt_selection(options: List[str]) -> int:
    terminal_menu = TerminalMenu(options, search_key=None)
    selected_index = terminal_menu.show()
    if selected_index is None:
        print('No selection made! Aborting...')
        exit()
    return selected_index    

def get_bearer_token(server_url: str, client_id: str, realm: str, client_secret: str):
    keycloak_openid = KeycloakOpenID(server_url=server_url,
                                 client_id=client_id,
                                 realm_name=realm,
                                 client_secret_key=client_secret,
                                 verify=False)
    token = keycloak_openid.token(client_id, client_secret, grant_type="client_credentials")
    return token['access_token']

def init_grpc_stub(grpc_url):
    channel = grpc.secure_channel(target=grpc_url,
                                  credentials=grpc.ssl_channel_credentials(
                                      root_certificates=None))

    stub = CameraController_pb2_grpc.CameraControllerStub(channel=channel)
    return stub

def fetch_cameras(token: str) -> List:
    headers = {"Authorization": "Bearer " + token}
    res = requests.get(f"{CAMERASERVICE_URL}/v1/cameras", headers=headers)

    if res.status_code != 200:
        print("Failed to get camera list from CameraService.", res)
        exit()

    return json.loads(res.content)['cameras']

def camera_name_for_id(cameras, id):
    matching_camera = next(filter(lambda c: c['id'] == str(id), cameras), None)
    return matching_camera['name'] if matching_camera is not None else None      
    
parser = argparse.ArgumentParser(description='Record stream from CameraService', epilog='All options that are not specified but necessary will be prompted for.')
parser.add_argument('-c', '--camera-id', type=int, help='Camera id within CameraService')
parser.add_argument('-t', '--record-length', type=int, help='Record length in seconds')
parser.add_argument('-l', '--log-frametimes', action='store_true', help='Log frametimes into a file `.frametimes`')
args = parser.parse_args()

keycloak_client_secret = os.getenv('KEYCLOAK_CLIENT_SECRET')
if keycloak_client_secret is None:
    print('(You can also supply the client secret via env var KEYCLOAK_CLIENT_SECRET)')
    keycloak_client_secret = getpass('Keycloak client secret: ')

token = get_bearer_token(server_url=AUTH_URL,
                            client_id=CLIENT_ID,
                            realm=REALM,
                            client_secret=keycloak_client_secret)

cameras = fetch_cameras(token)

camera_id = 10

camera_name = camera_name_for_id(cameras, camera_id)
file_name = f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}_id{camera_id}_{camera_name}"

grpc_stub = init_grpc_stub('grpc2-carmel.vw2-demospace.de:443')

request = StreamRequest(
    camera_id=camera_id,
    max_fps=100,
    size=50,
)

metadata = (('authorization', 'Bearer ' + token),)
responses = grpc_stub.Stream(request=request, metadata=metadata, timeout=None)
count = 0
start_time = time.time()
last_frame_time = time.time()

# Initialize YOLOv7 model
model_path = "models\yolov7_736x1280.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

try:
    for response in responses:
        frame = cv2.imdecode(np.asarray(bytearray(response.frame)),cv2.IMREAD_COLOR)
        boxes, scores, class_ids = yolov7_detector(frame)
        combined_img = yolov7_detector.draw_detections(frame)
        cv2.imshow("Detected Objects", combined_img)
        cv2.waitKey(1) 
except Exception as e:
    traceback.print_exc()