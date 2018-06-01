# Shape Recognition Challenge

## Run
To run the logic and hence pass the input image via command line to the main script use the following command:

```
python main.py -i <image path string>
```

## Quick evaluation
The approach taken consisted in using image processing using OpenCV, numpy and imutils as libraries. The logic first processed the image using several processing algorithms and then computes the contours of the blob in order to determine the probable class.

This approach is really weak when the object are close to each other, making it almost impossible to recognise correctly the distinct objects. Nonetheless, has proven to work well in most of the other cases even when the instances are small.

If I could improve I would probably use CNN + ANN and treat the problem as a multi class one as presented in one of the papers read when doing a bit of research for the challenge (Counting with CNNs using deep learning features).

Overall it was fun and interesting and I reckon I spent probably around 4/5 hours on and off for this basic implementation.
