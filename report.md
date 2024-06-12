
## Preprocessing

This part filters the video for frames showing the table in the desired perspective.

We noticed, that almost all target frames contain a scoreboard (the names on yellow background and the white and blue area) in the bottom of the frame. We handpicked the location of this scoreboard and compared this part of all frames to the sample frame. We include all frames in the filtered video for which difference of this part to the sample image is smaller than a handpicked threshold.

We observed some misclassifications, as for example in repeated video segments the score is not shown. Consequently, these segments are not contained in our filtered video. On the other hand, there are occasionally some trajectories drawn on the screen in white while the scoreboard is shown, so these frames are included. However, we suppose these cases are mere edge-cases and our filtering by the scoreboard works well enough. 



## Detect Highlight

To detect the image coordinates of the highlight, we use the detect_highlight function.

First, we select the pixels of the video frame that have the color of the balls. This gives us masks for each color of balls. We handpicked the ranges for each ball color. 
Our current selection yields some false positives of red balls on the pink ball, and the distinction of the white and yellow ball, as well as the detection of the black ball, are not as robust as we hoped for.

We then erode these masks to remove any noise (for example, the shadows of all balls have the same color als the black ball) and dilate them to make them big enough to encompass the highlight. We then only select the largest connected group of pixels that morphologically resembles a ball to further eliminate any noise that made it through the eroding step, such as the players wearing a black shirt walking into the frame. An exception to this step is the red mask. As there are multiple red balls we cannot select the largest group, so this entire step is skipped.

The next step is to find the highlights. We check for every pixel in a given mask if the above neighbour is bright enough to be a highlight. This yields us the lower edge of the highlights. We then dilate these masks such that they cover the whole specular. 

In these dilated masks we then select all bright pixels that we consider part of the highlight. We noticed, that the highlight has different colors for different balls, so we differentiate here. For example, the highlight on the black ball is much darker than on the others. 
Finally, using these highlight-pixels, we calculate the exact position of the reflection by getting the position of the lowest pixel as the y coordinate and the mena of the highlight pixels as the x coordinate. We chose to get the lower end of the reflection as this might be detectable more exactly than the middle. We handle the red balls in a similar fashion by just calculating the location for each group of pixels in the mask.