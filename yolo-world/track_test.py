from ultralytics import YOLOWorld

import argparse

parser= argparse.ArgumentParser(description=" YOlo v8 Word tracking test")
parser.add_argument('-t','--track',type=str, default=None,help="video input ")
parser.add_argument('-m','--model',type=str,default="yolov8m-worldv2.pt",help="videos you want to track on ")

args = parser.parse_args()


model = YOLOWorld(model = args.model)


results = model.track(source=args.track,show = True,device= 0, save = True)




