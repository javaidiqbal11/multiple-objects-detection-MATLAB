% Create System objects for reading and displaying video and for drawing bounding boxes around the detected people.
videoReader = vision.VideoFileReader('C:\Users\professionals\Downloads\MATLAB_Solution\sampl2.mp4');
videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);

% Initialize the people detector with default settings or customized settings.
peopleDetector = vision.PeopleDetector('MinSize', [166 83]);

% Process each frame of the video.
while ~isDone(videoReader)
    frame = step(videoReader);  % read the next video frame

    % Detect people in the frame.
    [peopleBoxes, scores] = step(peopleDetector, frame);

    % Filter detections by score (if necessary, adjust threshold according to your needs).
    strongDetections = scores > 0;  % Adjust score threshold if needed
    peopleBoxes = peopleBoxes(strongDetections, :);

    % Draw bounding boxes around the detected people.
    result = insertShape(frame, 'Rectangle', peopleBoxes, 'Color', 'green');

    % Display the number of people detected.
    numPeople = size(peopleBoxes, 1);
    result = insertText(result, [10 10], ['Detected People: ' num2str(numPeople)], 'BoxOpacity', 1, ...
        'FontSize', 14);

    step(videoPlayer, result);  % display the results
end

% Clean up
release(videoReader);
release(videoPlayer);
