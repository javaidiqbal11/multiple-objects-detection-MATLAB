% Define the URL of the smart camera
cameraUrl = 'http://192.168.x.x/video';  % Change this to your camera's URL
camera = ipcam(cameraUrl);

% Test connection by grabbing one frame
testFrame = snapshot(camera);
figure, imshow(testFrame), title('Test Snapshot from Camera');

% Create video player for displaying the video
videoPlayer = vision.DeployableVideoPlayer('Location', [100, 100]);

% Continue setting up as before
foregroundDetector = vision.ForegroundDetector('NumTrainingFrames', 10, ...
                                               'InitialVariance', 30*30);
blobAnalyzer = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
                                   'AreaOutputPort', true, ...
                                   'CentroidOutputPort', true, ...
                                   'MinimumBlobArea', 150);
particleFilter = vision.ParticleFilter('NumParticles', 100, ...
                                       'InitialEstimateError', [200, 50], ...
                                       'StateEstimationMethod', 'Mean', ...
                                       'StateTransitionFcn', @stateTransitionFcn, ...
                                       'MeasurementLikelihoodFcn', @measurementLikelihoodFcn);

% PID controller initialization
Kp = 0.1; Ki = 0.01; Kd = 0.05;
pidController = pid(Kp, Ki, Kd);

% Main loop to process frames from the camera
while isOpen(videoPlayer)
    frame = snapshot(camera);  % Capture frame from the smart camera
    foregroundMask = step(foregroundDetector, frame);
    [~, centroids, bboxes] = step(blobAnalyzer, foregroundMask);
    
    if ~isempty(centroids)
        initialize(particleFilter, centroids(1,:), [200, 50]);
    end
    
    predictedCentroid = predict(particleFilter);
    if ~isempty(centroids)
        correctedCentroid = correct(particleFilter, centroids(1,:));
    else
        correctedCentroid = predictedCentroid;  % No detection, rely on prediction
    end
    
    errorSignal = correctedCentroid - size(frame)/2;
    controlSignal = step(pidController, errorSignal);
    
    % Visualize and possibly adjust camera control based on error
    frame = insertShape(frame, 'Rectangle', bboxes, 'Color', 'green');
    frame = insertMarker(frame, correctedCentroid, 'x', 'Color', 'red');
    step(videoPlayer, frame);
end

% Clean up
clear('camera');
release(videoPlayer);
