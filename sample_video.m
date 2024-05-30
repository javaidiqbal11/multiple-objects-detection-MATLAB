% Create System objects for reading and displaying video and for drawing bounding boxes around the tracked objects.
videoReader = vision.VideoFileReader('sample.mp4');
videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 50, 'MinimumBackgroundRatio', 0.7);

blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', true, 'CentroidOutputPort', true, ...
    'MinimumBlobArea', 150);

% Initialize the people detector.
peopleDetector = vision.PeopleDetector('MinSize', [166 83]);

% Process each frame of the video.
while ~isDone(videoReader)
    frame = step(videoReader);  % read the next video frame

    % Use the foreground detector to identify moving objects in the video.
    foreground = step(foregroundDetector, frame);

    % Apply morphological operations to remove noise and fill in holes.
    cleanedForeground = imopen(foreground, strel('Disk', 1));
    cleanedForeground = imclose(cleanedForeground, strel('Disk', 15));
    cleanedForeground = imfill(cleanedForeground, 'holes');

    % Perform blob analysis to find connected components.
    [~, centroids, bboxes] = step(blobAnalyser, cleanedForeground);

    % Detect people in the frame.
    peopleBoxes = step(peopleDetector, frame);
    
    % Combine bounding boxes from blob analysis and people detection
    allBoxes = [bboxes; peopleBoxes];

    % Draw bounding boxes around the detected objects.
    result = insertShape(frame, 'Rectangle', allBoxes, 'Color', 'green');

    % Display centroids
    result = insertMarker(result, centroids, 'x', 'Color', 'red');

    % Display the number of objects being tracked.
    numObjects = size(centroids, 1) + size(peopleBoxes, 1);
    result = insertText(result, [10 10], ['Total Objects: ' num2str(numObjects)], 'BoxOpacity', 1, ...
        'FontSize', 14);

    step(videoPlayer, result);  % display the results
end

% Clean up
release(videoReader);
release(videoPlayer);
