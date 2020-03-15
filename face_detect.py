from flask import Flask, jsonify, request
import json

import asyncio
import io
import glob
import os
import secure
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType

app = Flask("Connexus")

## NOTE: Need secure.py file from localhost, will not run otherwise
KEY = secure.KEY
ENDPOINT = secure.ENDPOINT
PERSON_GROUP_ID = 'twentieth-batch'
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# def load_client():
#     face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
#     return face_client


def detect_faces(file_name):
    image = open(file_name, 'r+b')
    # Detect faces
    face_ids = []
    faces = face_client.face.detect_with_stream(image)
    if not faces:
        raise Exception('No face detected from image.')
    for face in faces:
        face_ids.append(face.face_id)
    image.close()
    return faces, face_ids


def get_rectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height

    return ((left, top), (right, bottom))


def show_image(file_name, detected_faces=[], confidence_vals=None):
    data = open(file_name, 'rb')
    img = Image.open(data)

    # For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    # draw.text((x, y),"Sample Text",(r,g,b))
    if detected_faces == []:
        # w, h = img.size
        # font = ImageFont.load_default()
        # text_w, text_h = draw.textsize(title, font)
        # draw.text(((w - text_2) // 2, h - text_h), "We have confirmed this child as a match with {0:.{1}f}% confidence".format(confidence_vals[0]*100, 1), (255, 255, 255))
        print("We have confirmed this child as a match with {0:.{1}f}% confidence".format(confidence_vals[0]*100, 1))
    for face in detected_faces:
        draw.rectangle(get_rectangle(face), outline='red')

    # Display the image in the users default image browser.
    img.show()


def create_person_group(names):
    print('Person group:', PERSON_GROUP_ID)
    face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)
    name_map = {}
    for name in names:
        name_map[name] = face_client.person_group_person.create(PERSON_GROUP_ID, name)
    return name_map


def train_person_group(names, name_map):
    for name in names:
        images = [file for file in glob.glob('*.png') if file.startswith(name)]
        for image in images:
            data = open(image, 'r+b')
            face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, name_map[name].person_id, data)
    print('Training the person group...')
    # Train the person group
    face_client.person_group.train(PERSON_GROUP_ID)

    while (True):
        training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
        print("Training status: {}.".format(training_status.status))
        print()
        if (training_status.status is TrainingStatusType.succeeded):
            break
        elif (training_status.status is TrainingStatusType.failed):
            sys.exit('Training the person group has failed.')
        time.sleep(5)

def identify_child( file_name, names, name_map):
    faces, face_ids = detect_faces(file_name)
    show_image(file_name, faces)
    train_person_group(names, name_map)
    results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
    confidence_vals = []
    for person in results:
        print(person.candidates[0].confidence)
        confidence_vals.append(person.candidates[0].confidence)
    return results, confidence_vals


if __name__ == '__main__':
    names = ["blue-ivy", "zahara", "eddie", "blackish"]
    test_image = "test-blackish.png"
    # Step 0: Load face client
    #face_client = load_client()

    # Step 1: Create person_group with correct pictures
    name_map = create_person_group(names=names)

    # # Step 2: Train person group on correct images
    # train_person_group(names, name_map)

    # Step 3: Identify from test image
    results, confidences = identify_child(file_name=test_image, names=names, name_map=name_map)

    # Step 4: Return confidence level
    print(confidences)

    for key in name_map:
        if name_map[key].person_id == results[0].candidates[0].person_id:
            print(key)
            images = [file for file in glob.glob('*.png') if file.startswith(key)]
            for image in images[:1]:
                show_image(image, confidence_vals=confidences)



