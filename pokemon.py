#!/usr/bin/python3
from mlforkids import MLforKidsImageProject

# treat this key like a password and keep it secret!
key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"x

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction("/path/to/test.jpg")

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))
