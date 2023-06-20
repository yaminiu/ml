#!/usr/bin/python3
from mlforkids import MLforKidsImageProject

# treat this key like a password and keep it secret!
key = "718280d0-0f5c-11ee-8bb1-1f569cdc22a4a5f7362d-0020-41aa-ae1f-b533f100ad79"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction("/mnt/c/Users/ethan/OneDrive/Desktop/Ethandoc/icode/pokemon2.jpg")

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))