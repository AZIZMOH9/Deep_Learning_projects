import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sort import *
import torch
from super_gradients.training import models


device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#yolo model initialize
model=models.get("yolo_nas_s", pretrained_weights='coco').to(device)
cap=cv2.VideoCapture('Video4.mp4')
#hight and width of the video
w=int(cap.get(3))
h=int(cap.get(4))
#initializeing the tracker

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
#the lines cordinat in the video
down_line=[225,850,963,850]
up_line=[979,850,1667,850]


#lists to store the ides of entering and outgoing cars


down_count=[]
up_count=[]

out=cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','J'),10,(w,h))


while (True):
    ret,frame=cap.read()

    if (ret):
        #making predection and extract the information from the predction
        detections=np.empty((0,6))
        predict=list(model.predict(frame,conf=0.35))[0]
        bbx=predict.prediction.bboxes_xyxy.tolist()
        confidenc=predict.prediction.confidence
        labels=predict.prediction.labels.tolist()

        for (bbs,confs,label) in zip(bbx,confidenc,labels):
            x1,y1,x2,y2=int(bbs[0]),int(bbs[1]),int(bbs[2]),int(bbs[3])
            conf=math.ceil((confs*100))/100
            label=int(label)
            current_array=np.array([x1,y1,x2,y2,conf,label])
            detections=np.vstack((detections,current_array))
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
        #update the tracker information
        track_det=tracker.update(detections)

        if len(track_det)>0:
            #extracting information from the tracker
            bbx = track_det[:, :4]
            category = track_det[:, 4]
            identity = track_det[:, 8]
            cv2.line(frame,(down_line[0],down_line[1]),(down_line[2],down_line[3]),(0,0,0),6)
            cv2.line(frame, (up_line[0], up_line[1]), (up_line[2], up_line[3]), (0, 0, 0), 6)
            for i, box in enumerate(bbx):
                offset=(0,0)
                x1, y1, x2, y2 = [int(i) for i in box]
                x1 += offset[0]
                y1 += offset[0]
                x2 += offset[0]
                y2 += offset[0]
                id = int(identity[i])
                #label = str(id) + ":" + str(cat)
                cx,cy=int((x1+x2)/2),int((y1+y2)/2)
                cv2.circle(frame,(cx,cy),3,(255,0,0),-1)
                # (w,h),_ = cv2.getTextsize(label , cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                cv2.rectangle(frame,(x1,y1),(x1+32,y1-20),(0,66,200),-1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if down_line[0] <cx<down_line[2] and down_line[1]-20 <cy< down_line[3]+20:
                    if (down_count.count(id)==0):
                        cv2.line(frame, (down_line[0], down_line[1]), (down_line[2], down_line[3]), (0, 0, 255), 6)
                        down_count.append(id)

                if up_line[0] <cx<up_line[2] and up_line[1]-20<cy<up_line[3]+20:
                    if (up_count.count(id)==0):
                        cv2.line(frame, (up_line[0], up_line[1]), (up_line[2], up_line[3]), (0, 0, 255), 6)
                        up_count.append(id)

                cv2.rectangle(frame, (1200,40), (1730, 150), (30, 30, 250), -1)
                cv2.rectangle(frame, (75, 40), (650, 150), (30, 30, 250), -1)
                cv2.putText(frame, str(id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)
        sk = "the Count of entered car:  " + str(len(up_count))
        sf = "the Count of outgoing car:  " + str(len(down_count))
        cv2.putText(frame, str(sk), (1230, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 3)
        cv2.putText(frame, str(sf), (105, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 3)
        out.write(frame)
        cv2.imshow("frmae",frame)
        key=cv2.waitKey(1)
        if (key)==97:
            break


    else:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
