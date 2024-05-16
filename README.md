# 3D Ants
Extrapolating Segments with 3d Ants

Previously I did some work that worked in 2d on the scroll volume to try to extend or clean up segmentations. The formulation of this was to give some history of points from an existing obj file for a single tif layer and then try to predict the next x, y coordinate or the movement from last known points. This could predict just the next point or it could predict the next 15 points. Doing this it could autoregressively try to continue the segmentation further into the future. Learning from the millions of points that the annotators have already labeled by hand. This work was initially inspired by the Lyft motion prediction competition where we were given a history of points for a car and then tried to extrapolate where it was going to go. Outputting the next 30 timesteps of the cars path on the road. https://www.kaggle.com/code/corochann/lyft-deep-into-the-l5kit-library

This was reasonably successful but it was difficult to make it coherent across layers. The model might pick one path on layer 7000 and grab a different sheet on 7001 and after many extrapolations would completely diverge yielded a big mess of incoherent points. 

In order to solve this I have tried to extend it to 3d in various ways so the model is trying to output all of the layers at once in a chunk instead of one layer at a time independently.

The general premise of this modeling effort is:
1. Give the model a 256^3 volume centered on a point of papyrus
2. Mark history with a breadcrumb trail showing where segmentation came from
3. Orient/rotate the volume so that based on the last two known points forward is the direction the future will generally head. Similar to ego-centering in the self-driving car case. Makes the pattern much easier to learn when the model knows generally to trace forward and just veer left or right sometimes. Predicting in all directions much harder
4. Pass this input to the model and make it try to predict the points
5. As an auxiliary target try to predict the drawn out segmentation line, this aids in learning and can potentially be used directly with some ingenuity on how to resolve this back to points

**Segmentation prediction and label**

![Segmentation prediction](https://github.com/ryanchesler/ants/blob/main/prediction_segmentation.gif?raw=true)
![Segmentation label](https://github.com/ryanchesler/ants/blob/main/label_segmentation.gif?raw=true)

I have tried various methods around this with different characteristics

**Variant 1: Model both directions**

 - Predict the future but also the history points
 - History points should be very easy to locate because those are directly shown with a breadcrumb in the volume, almost like a baby object detection problem, just trying to learn to trace the points we planted on the volume
 - 15 points history, 15 points future
 - 256 layer * 30 point per layer output
 - Easy future inference pattern because we can just scooch along one point at a time into the future and rerun inference
 - One of the struggles with this approach is the center layer is easy to predict because it is always centered, but the layers above and below drift away from the center so the model ends up needing to learn not only the forecast but also the offset because the points wont all be neatly stacked up. Often the models predictions will actually be reasonable but it is penalized because its point 10 might be better aligned with point 25 in the sequence or something like that


**Variant 2:Only forward**

Same as above but only predicting forward points, not predicting history


**Variant 3:Predicting all**

Please ignore the weirdness at the end of extrapolation, just a plotting bug when points reach off the sheet. Not a very extensively trained model but shows the capability. There are millions and millions of points to crop and train against and only been trained against a fraction of a percent

![Predicting all](https://github.com/ryanchesler/ants/blob/main/full_point.gif?raw=true)


 - Instead of only predicting a fixed 15 pts history, 15 points future, predict all points visible on all layers
 - Align the first predicted point for every z layer to be its entrance to the volume and have the model predict every point until it leaves the volume
 - Very strong modeling wise, can even just put down a single central point and have it try to extrapolate full history and future for all layers
 - Not incorrectly penalized for the ordering of points or offset of points because it knows it has to trace to the beginning and end of the sheet every time
 - Difficult inference pattern because what is defined by future every time is going to be slightly different. It could be any ordering of the point that switches from history to future


Metric plot showing how performance varies with layers, x is layer number and y is average euclidean distance for that layer. Middle 128 layer performs the strongest because this is the point the whole volume is centered on, model has it easy because it always knows it will pass through 128,128. The middle layer only being off by an average of 6 pixels is very strong. All the way at the ends being only off by 14 pixels is still good, but probably could be significantly improved with more training

![Layer plot](https://github.com/ryanchesler/ants/blob/main/layer%20metrics_56_564c6d344bd3c0bc0740.png?raw=true)

Metric plot showing how performance changes as you extrapolate further into the future. X is the predicted timestep and y is the average euclidean distance. Again we can see that performance is actually very strong, especially in earlier timesteps. All the way at the end of the extrapolation is probably significantly worse because many of the segments simply dont go that far so those further out neurons are undertrained. We can easily cap the extrapolation distance it shift and rerun inference with a recentered volume instead of trying to extrapolate all the way off the page

![Extrapolation plot](https://github.com/ryanchesler/ants/blob/main/extrapolation%20metrics_56_0a3a9062448485968152.png?raw=true)

**Variant 4:Segmentation Only**

Skipping all point regression and just trying to output the segmentation line, potentially useful because we can take this and resolve back down to points. This is the most easily learnable pattern, but requires some postprocessing to convert to points again. This is very powerful and differs from other approaches that have tried instance segmentation and generic sheet segmentation because it will specifically highlight the sheet that you have centered on and try to steer a single line through it. Other approaches have been questionably useful because they dont directly put us closer to producing new objs, require way more steps to get there. 
