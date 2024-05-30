% Step 1: Setup Camera Connection
camera = webcam; % Connect to the camera
preview(camera); % Open a video preview window

% Step 2: Background Subtraction for Object Detection
background = rgb2gray(snapshot(camera)); % Take initial background frame
figure; % Create a new figure for processing output
hold on;

while true
    frame = snapshot(camera); % Take a current snapshot from camera
    grayFrame = rgb2gray(frame); % Convert frame to grayscale for processing
    
    % Calculate the absolute difference between the background and current frame
    difference = abs(double(grayFrame) - double(background));
    
    % Threshold the difference to get binary image of moving objects
    threshold = 30; % Threshold value might need adjustment
    binaryImage = imbinarize(difference, threshold / 255);
    
    % Apply morphological operations to remove noise and fill gaps
    binaryImage = bwareaopen(binaryImage, 50); % Remove small objects
    binaryImage = imclose(binaryImage, strel('disk', 15)); % Close gaps
    
    % Find boundaries of moving objects
    [boundaries, ~] = bwboundaries(binaryImage, 'noholes');
    
    % Display the original frame
    imshow(frame);
    hold on;
    
    % Step 3: Highlight Detected Objects
    for k = 1:length(boundaries)
        boundary = boundaries{k};
        plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2); % Draw boundaries in green
    end
    
    hold off;
    
    % Update the background (simple running average)
    % This could be improved with a more complex background update mechanism
    background = (0.9 * double(background) + 0.1 * double(grayFrame));
    
    % Pause for demonstration clarity
    pause(0.05);
end

% Clean up
clear('camera'); % Release the camera
close all; % Close figures
