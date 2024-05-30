% Add required toolboxes
assert(exist('vision.ForegroundDetector', 'class') == 8, 'Computer Vision Toolbox is required.');
assert(exist('vision.ParticleFilter', 'class') == 8, 'Computer Vision Toolbox is required.');

% Create objects for reading and playing video
videoFile = 'tester.mp4';
videoReader = vision.VideoFileReader(videoFile, 'VideoOutputDataType', 'uint8');
videoPlayer = vision.DeployableVideoPlayer('Location', [100, 100]);

% Create foreground detector (Background subtraction)
foregroundDetector = vision.ForegroundDetector('NumTrainingFrames', 10, ...
                                               'InitialVariance', 30*30);

% Create blob analyzers
blobAnalyzer = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
                                    'AreaOutputPort', true, ...
                                    'CentroidOutputPort', true, ...
                                    'MinimumBlobArea', 150);

% Initialize Particle Filter for the tracker
particleFilter = vision.ParticleFilter('NumParticles', 100, ...
                                       'InitialEstimateError', [200, 50], ...
                                       'StateEstimationMethod', 'Mean', ...
                                       'StateTransitionFcn', @stateTransitionFcn, ...
                                       'MeasurementLikelihoodFcn', @measurementLikelihoodFcn);
                                       
% PID controller to adjust camera based on position error
Kp = 0.1; Ki = 0.01; Kd = 0.05;
pidController = pid(Kp, Ki, Kd);


while ~isDone(videoReader)
    frame = step(videoReader);
    % Detect the foreground
    foregroundMask = step(foregroundDetector, frame);
    
    % Analyze connected components to find measurements for tracking
    [~, centroids, bboxes] = step(blobAnalyzer, foregroundMask);
    
    % If detection exists, initialize or update particle filter
    if ~isempty(centroids)
        initialize(particleFilter, centroids(1,:), [200, 50]);
    end
    
    % Predict and correct particle filter based on measurements
    predictedCentroid = predict(particleFilter);
    if ~isempty(centroids)
        correctedCentroid = correct(particleFilter, centroids(1,:));
    else
        correctedCentroid = predictedCentroid;  % No detection, rely on prediction
    end
    
    % PID controller to adjust camera view based on centroid error
    errorSignal = correctedCentroid - size(frame)/2;
    controlSignal = step(pidController, errorSignal);
    
    % Modify camera view or simulate control output
    % (This part depends on your camera control system)
    
    % Visualize tracking
    frame = insertShape(frame, 'Rectangle', bboxes, 'Color', 'green');
    frame = insertMarker(frame, correctedCentroid, 'x', 'Color', 'red');
    step(videoPlayer, frame);
end



release(videoReader);
release(videoPlayer);
