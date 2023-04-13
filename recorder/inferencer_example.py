import argparse
import json
import os
import time
from datetime import datetime, timezone
from getpass import getpass
import traceback
from typing import Callable, List

import cv2
import grpc
import numpy as np
import requests
from icv.camera.v1 import CameraController_pb2_grpc
from icv.camera.v1.CameraController_pb2 import GetImageRequest, StreamRequest
from keycloak import KeycloakOpenID
from simple_term_menu import TerminalMenu

from ultralytics import YOLO

# access info
CLIENT_ID = "purdue"
REALM = "icv"

# VW2 coordinates
BASE_PLATFORM_URL="https://carmel2.vw2-demospace.de/carmel-smart-cities-"
CAMERASERVICE_URL=f"{BASE_PLATFORM_URL}cameraservice"
AUTH_URL=f"{BASE_PLATFORM_URL}auth/"

def main():
    parser = argparse.ArgumentParser(description='Record stream from CameraService', epilog='All options that are not specified but necessary will be prompted for.')
    parser.add_argument('-c', '--camera-id', type=int, help='Camera id within CameraService')
    parser.add_argument('-t', '--record-length', type=int, help='Record length in seconds')
    parser.add_argument('-l', '--log-frametimes', action='store_true', help='Log frametimes into a file `.frametimes`')
    args = parser.parse_args()

    token = get_bearer_token(server_url=AUTH_URL,
                                client_id=CLIENT_ID,
                                realm=REALM)

    cameras = fetch_cameras(token)
    camera_id, record_length = get_cam_and_length(args, cameras)
    camera_name = camera_name_for_id(cameras, camera_id)

    file_name = f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}_id{camera_id}_{camera_name}"
    log_frametimes = args.log_frametimes
    log_file = None
    if log_frametimes:
        log_file = open(f'{file_name}.frametimes', 'w')
        f_logger = create_frametime_logger(log_file)

    print(f'About to start inferencing {record_length}s from camera with id {camera_id} ({camera_name}).')

    grpc_stub = init_grpc_stub('grpc2-carmel.vw2-demospace.de:443')

    request = StreamRequest(
        camera_id=camera_id,
        max_fps=100,
        size=100,
    )
    metadata = (('authorization', 'Bearer ' + token),)
    responses = grpc_stub.Stream(request=request, metadata=metadata, timeout=None)
    count = 0
    start_time = time.time()
    last_frame_time = time.time()

    #setup yolo
    model = YOLO("yolov8s.pt")

    video_writer = cv2.VideoWriter(f'{file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (1920, 1080))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 800, 600)

    try:
        # called whenever a new frame arrives
        for response in responses:
            frame_time = round(time.time() - last_frame_time, 5)
            print(f"Frametime: {frame_time}s")
            if log_frametimes:
                f_logger(frame_time)
            last_frame_time = time.time()
            frame = cv2.imdecode(np.asarray(bytearray(response.frame)),cv2.IMREAD_COLOR)
            prediction = model.predict(frame)
            res_plotted = prediction[0].plot()
            video_writer.write(res_plotted)
            cv2.imshow('output', res_plotted)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break 
            count += 1
            if time.time() - start_time > record_length:
                break
    except Exception as e:
        traceback.print_exc()
    finally:
        video_writer.release()
        total_time = time.time() - start_time
        print(f"Retrieved {count} frames in {round(total_time, 2)}s. Avg.: {round(count/total_time, 2)}fps")
        print(f"Wrote video output to {file_name}.avi")
        if log_file is not None:
            print(f'Logged frame times into {log_file.name}')
            log_file.close()
        cv2.destroyAllWindows()


def get_keycloak_secret():
    keycloak_client_secret = os.getenv('KEYCLOAK_CLIENT_SECRET')
    if keycloak_client_secret is None:
        print('(You can also supply the client secret via env var KEYCLOAK_CLIENT_SECRET)')
        keycloak_client_secret = getpass('Keycloak client secret: ')
    return keycloak_client_secret

def get_bearer_token(server_url: str, client_id: str, realm: str):

    client_secret = os.getenv('KEYCLOAK_CLIENT_SECRET')
    if client_secret is None:
        print('(You can also supply the client secret via env var KEYCLOAK_CLIENT_SECRET)')
        client_secret = getpass('Keycloak client secret: ')

    keycloak_openid = KeycloakOpenID(server_url=server_url,
                                 client_id=client_id,
                                 realm_name=realm,
                                 client_secret_key=client_secret,
                                 verify=False)
    token = keycloak_openid.token(client_id, client_secret, grant_type="client_credentials")
    return token['access_token']

def get_cam_and_length(args, cameras):
    camera_id = args.camera_id
    if camera_id is None:
        selection_index = prompt_selection(map(lambda c: c['name'], cameras))
        camera_id = int(cameras[selection_index]['id'])

    record_length = args.record_length
    if record_length is None:
        record_length = prompt_number(default=10)
    
    return camera_id, record_length

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

def init_grpc_stub(grpc_url):
    channel = grpc.secure_channel(target=grpc_url,
                                  credentials=grpc.ssl_channel_credentials(
                                      root_certificates=None))

    stub = CameraController_pb2_grpc.CameraControllerStub(channel=channel)
    return stub

def create_frametime_logger(file) -> Callable:
    def logger(msg):
        file.write(f'{msg}\n')
    return logger

def prompt_selection(options: List[str]) -> int:
    terminal_menu = TerminalMenu(options, search_key=None)
    selected_index = terminal_menu.show()
    if selected_index is None:
        print('No selection made! Aborting...')
        exit()
    return selected_index

def prompt_number(default: int) -> int:
    try:
        response = input(f'Record length in s [{default}]: ').strip()
        return int(response) if len(response) > 0 else default
    except ValueError:
        print('Invalid number! Aborting...')
        exit()

main()