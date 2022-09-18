import cv2 as cv
import os
from matplotlib import pyplot as plt

base_path = 'Dataset/images'

inter_image = cv.imread('Dataset/target.jpg')
inter_image = cv.cvtColor(inter_image, cv.COLOR_BGR2GRAY)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))

scene_images = []
for i in os.listdir(base_path):
    file = i.split('.') 
    if file[1] == 'jpg':
        image_path = cv.imread(base_path + '/' + i)
        image_path = cv.cvtColor(image_path, cv.COLOR_BGR2GRAY)
        image_path = clahe.apply(image_path)
        scene_images.append(image_path)

SURF = cv.xfeatures2d.SURF_create()

#kp + ds inter image
inter_kp, inter_ds = SURF.detectAndCompute(inter_image, None)

KDTREE_INDEX = 1 # 0 1 for SIFT SURF, 5 dipake untuk ORB
TREE_CHECKS = 50

FLANN = cv.FlannBasedMatcher(dict(algorithm = KDTREE_INDEX), dict(checks = TREE_CHECKS))

all_mask = []
scene_index = -1
total_match = 0
scene_keypoints = None
final_match = None

for index, i in enumerate(scene_images):
    scene_kp, scene_ds = SURF.detectAndCompute(i, None)
    matcher = FLANN.knnMatch(inter_ds, scene_ds, 2)
    match_count = 0
    scene_mask = [[0,0] for j in range(0, len(matcher))]

    for j, (m,n) in enumerate(matcher):
        if m.distance < 0.7 * n.distance:
            scene_mask[j] = [1,0]
            match_count +=1

    all_mask.append(scene_mask)
    if total_match < match_count:
        total_match = match_count
        scene_index = index
        scene_keypoints = scene_kp
        final_match = matcher

result = cv.drawMatchesKnn(
    inter_image, inter_kp,
    scene_images[scene_index], scene_keypoints,
    final_match, None, matchColor=[0,255,0],
    singlePointColor=[255,0,255],
    matchesMask=all_mask[scene_index]
)

plt.imshow(result, cmap="gray")
plt.show()