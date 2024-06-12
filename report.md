# Preprocessing

This part filters the video for frames showing the table in the desired perspective.

We noticed, that almost all target frames contain a scoreboard (the names on yellow background and the white and blue area) in the bottom of the frame. We handpicked the location of this scoreboard and compared this part of all frames to the sample frame. We include all frames in the filtered video for which difference of this part to the sample image is smaller than a handpicked threshold.

We observed some misclassifications, as for example in repeated video segments the score is not shown. Consequently, these segments are not contained in our filtered video. On the other hand, there are occasionally some trajectories drawn on the screen in white while the scoreboard is shown, so these frames are included. However, we suppose these cases are mere edge-cases and our filtering by the scoreboard works well enough.


## Static analysis of sample frame

From the sample frame we extract known features with known world coordinates, namely: the green to brown edges of the playing area, the baulk line, and the center of the yellow brown green and blue balls. We excluded the pink, white and black balls due to concerns in color filtering.


### Table edges and baulk line

To detect these features, we relied heavily on knowing their precise color ranges. By examining the sample frame on a image processing program, we handpicked the ranges in HSL colorspace. To detect the green/grown edge, we used small dilations of the regions of both colors to force an overlap. Once we obtained a binary image representing the lines we desired, we used our customized Hough line transform to obtain the theta angle and rho distance form the origin of the lines.


### Ball centers

Due to some of the balls being completely unoccluded in the sample frame, we decided to detect their centers. To compute the center of the ball, again we filtered for a specific range but only on the hue channel. This yielded a very precise contour of the balls, as the playing area is very consistent in hue. Then we applied an erosion using as kernel roughly the same shape as the balls themselves, but slightly smaller. This resulted in their centers.


### DLT

Given the previous features correspondences of image coordinates and world coordinates, we computed the camera matrix using the DLT algorithm. Both as specified in the theory slides and the normalized DLT algorithm. This latter one resulted in more precise results. From this step we also computed the camera position in world coordinates.


## Detect Highlight

To detect the image coordinates of the highlight for each frame of the video, we use the `detect_highlight` function.

First, we select the pixels of the video frame that have the color of the balls. This gives us masks for each color of balls. We handpicked the ranges for each ball color.
Our current selection yields some false positives of red balls on the pink ball, and the distinction of the white and yellow ball, as well as the detection of the black ball, are not as robust as we hoped for.

We then erode these masks to remove any noise (for example, the shadows of all balls have the same color as the black ball) and dilate them to make them big enough to encompass the highlight. We then only select the largest connected group of pixels that morphologically resembles a ball to further eliminate any noise that made it through the eroding step, such as the players wearing a black shirt walking into the frame. An exception to this step is the red mask. As there are multiple red balls we cannot select the largest group, so this entire step is skipped.

The next step is to find the highlights. We check for every pixel in a given mask if the above neighbour is bright enough to be a highlight. This yields us the lower edge of the highlights. We then dilate these masks such that they cover the whole specular.

In these dilated masks we then select all bright pixels that we consider part of the highlight. We noticed, that the highlight has different colors for different balls, so we differentiate here. For example, the highlight on the black ball is much darker than on the others.
Finally, using these highlight-pixels, we calculate the exact position of the reflection by getting the position of the lowest pixel as the y coordinate and the mena of the highlight pixels as the x coordinate. We chose to get the lower end of the reflection as this might be detectable more exactly than the middle. We handle the red balls in a similar fashion by just calculating the location for each group of pixels in the mask.


## Wold coordinates of balls

From the image coordinates of the highlights, we computed the world coordinates of the center of the balls by solving the geometry problem of the specular light reflection. This proved to be a bit imprecise and very sensitive to noise in the DLT. We generated a symbolic expression that related the position of the center of the ball with the distance of the Highlight to the camera in world coordinates. We solved this non-linear equation using a module from `scipy`, as we are only interested in machine precision solutions.

We spent a lot of time on this step, as it was sensitive to various parameters and we could not achieve very good results. First, it was dependent on the light world coordinates. To obtain a good approximation on the light world coordinates we computed the reflected rays of light from the camera in the sample frame. If these 3D rays from the camera where to converge with some error, we could use a least square solver to find the point closer to all lines. But the lines did not meaningfully converge. Reasons for this include noise in the highlight, the fact that the light is not a single point in space, diffusion of the specular reflection, inconsistencies detecting the same origin point of the light.

Instead of computing the light position, we obtained an "educated guess". This worked remarkably better.
