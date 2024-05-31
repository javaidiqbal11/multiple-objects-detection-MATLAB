% Initialize webcam
cam = webcam;
numFrames = 0;
maxFrames = 40;
background = [];
particles = [];
numParticles = 100;
particleWeights = ones(numParticles, 1) / numParticles;

figure;
while true
    % Capture frame
    frame = snapshot(cam);
    frame = im2double(frame);
    
    % Update background model
    if isempty(background)
        background = frame;
    end
    
    numFrames = numFrames + 1;
    if numFrames <= maxFrames
        background = (background * (numFrames - 1) + frame) / numFrames;
    end
    
    % Extract foreground mask
    foregroundMask = abs(frame - background) > 0.1;
    
    % Convert foreground mask to grayscale if needed
    foregroundMask = rgb2gray(im2double(foregroundMask));
    
    % Apply segmentation
    segmentedMask = imbinarize(foregroundMask, 0.5);
    segmentedMask = bwareaopen(segmentedMask, 50);
    
    % Blob tracking
    blobStats = regionprops(segmentedMask, 'BoundingBox', 'Centroid');
    imshow(frame);
    hold on;
    
    for i = 1:length(blobStats)
        bbox = blobStats(i).BoundingBox;
        centroid = blobStats(i).Centroid;
        
        % Display tracking results
        rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
        plot(centroid(1), centroid(2), 'b*');
        
        % Particle Filter tracking
        % Create particles around the detected object
        if isempty(particles)
            particles = repmat(centroid, numParticles, 1) + randn(numParticles, 2) * 20;
        end
        
        % Extract color histogram of tracked model
        trackedColorHist = calculateColorHistogram(frame, bbox);
        
        % Find histogram at particles locations
        particleHistograms = findHistogramsAtParticleLocations(frame, particles);
        
        % Calculate distance between histograms
        distances = calculateHistogramDistances(trackedColorHist, particleHistograms);
        
        % Find log likelihood
        logLikelihood = calculateLogLikelihood(distances);
        
        % Inverse transform sampling
        [updatedParticles, valid] = inverseTransformSampling(logLikelihood, particles, numParticles);
        if valid
            particles = updatedParticles;
        end
        
        % Estimate position from particles
        estimatedPosition = estimatePositionFromParticles(particles);
        
        % Calculate error position (X, Y) - center
        errorPosition = calculateErrorPosition(estimatedPosition, centroid);
        
        % Control motors (pseudo code for controlling camera)
        controlMotors(errorPosition);
    end
    
    hold off;
    
    % Check for termination condition (e.g., key press)
    if ~isempty(get(gcf, 'CurrentCharacter'))
        break;
    end
end

% Cleanup
clear cam;

% Function definitions
function hist = calculateColorHistogram(frame, bbox)
    % Extract color histogram of the region defined by bbox in the frame
    region = imcrop(frame, bbox);
    hist = imhist(region);
end

function histograms = findHistogramsAtParticleLocations(frame, particles)
    % Find histograms at the particle locations
    histograms = [];
    for i = 1:size(particles, 1)
        bbox = [particles(i, :) 20 20];  % Assuming 20x20 window for particle histograms
        histograms = [histograms; calculateColorHistogram(frame, bbox)];
    end
end

function distances = calculateHistogramDistances(hist1, histograms)
    % Calculate the distances between histograms
    distances = sum((histograms - hist1').^2, 2);
end

function logLikelihood = calculateLogLikelihood(distances)
    % Calculate the log likelihood based on distances
    logLikelihood = -0.5 * distances;
end

function [updatedParticles, valid] = inverseTransformSampling(logLikelihood, particles, numParticles)
    % Perform inverse transform sampling to update particles
    weights = exp(logLikelihood - max(logLikelihood));
    weights = weights / sum(weights);
    cdf = cumsum(weights);
    cdf = cdf / cdf(end);
    
    % Ensure uniqueness and handle cases with insufficient unique indices
    [cdf, uniqueIdx] = unique(cdf, 'stable');
    
    if length(uniqueIdx) < numParticles
        % Not enough unique indices, return invalid flag
        updatedParticles = particles;
        valid = false;
        return;
    end
    
    % Ensure indices do not exceed bounds
    jitter = rand(size(cdf)) * 1e-10;
    cdf = cdf + jitter;
    
    indices = interp1(cdf, 1:length(cdf), rand(numParticles, 1), 'nearest', 'extrap');
    indices = min(max(indices, 1), length(uniqueIdx));  % Ensure indices are within bounds
    
    validIndices = uniqueIdx(indices);
    validIndices = validIndices(validIndices <= size(particles, 1)); % Ensure indices do not exceed particle array size
    updatedParticles = particles(validIndices, :) + randn(length(validIndices), 2) * 5;  % Adding noise
    valid = true;
end

function estimatedPosition = estimatePositionFromParticles(particles)
    % Estimate the position from the particles
    estimatedPosition = mean(particles);
end

function errorPosition = calculateErrorPosition(estimatedPosition, centroid)
    % Calculate the error between the estimated position and the centroid
    errorPosition = estimatedPosition - centroid;
end

function controlMotors(errorPosition)
    % Control motors based on error position (pseudo code)
    % Move camera to reduce the errorPosition
    % This is placeholder code and should be replaced with actual motor control code
    disp(['Error Position: ' num2str(errorPosition)]);
end
