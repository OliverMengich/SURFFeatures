%%
I1 = imread('IMG_20200123_170135_8.jpg');
I2 = imread('SceneImage1.jpg');
%imshow(I1)
%imshow(I2)
I1 = rgb2gray(I1);
I2 = rgb2gray(I2);
%%
points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);
%%
%imshow(I1); hold on; plot(points1.selectStrongest(10));
%imshow(I2); hold on; plot(points2.selectStrongest(100));
%%
[feats1,validpts1]= extractFeatures(I1,points1);
[feats2,validpts2] = extractFeatures(I2,points2);

%%
figure; imshow(I1);hold on; plot(validpts1,'showOrientation',true);
title('Detected Features');

%% Match features
index_pairs = matchFeatures(feats1,feats2,'Prenormalized',true);
matched_pts1 = validpts1(index_pairs(:,1));
matched_pts2 = validpts2(index_pairs(:,2));
figure; showMatchedFeatures(I1,I2,matched_pts1,matched_pts2,'montage');
title('Initial matches');

%% Remove outliers while estimating geometric transform using RANSAC function
[tform,inlierpoints1,inlierpoints2]= estimateGeometricTransform(matched_pts1,matched_pts2,'affine');
figure; showMatchedFeatures(I1,I2,inlierpoints1,inlierpoints2,'montage'); 
title('Filtered matches');
%%
boxpolygon = [1 1;...
    size(I1,2),1;...
    size(I1,2),size(I1,1);...
    1,size(I1,1);...
    1,1];


%% use estimate trsnsfer to locate object
newBoxPolygon = transformPointsForward(tform,boxpolygon);
figure; imshow(I2); hold on;
line(newBoxPolygon(:,1),newBoxPolygon(:,2),'Color','r','LineWidth',2);
title('Detected Object');
