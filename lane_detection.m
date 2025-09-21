% lane_detection.m
% Simple lane detection demo using Canny + Hough + polynomial/linear fit
% Requirements: MATLAB, Image Processing Toolbox
clear; clc; close all;

videoFile = 'road_video.mp4';   % put your video file here
outputFile = 'lane_output.avi'; % output video

% Parameters (tune for your video)
gaussSigma = 2;                % gaussian smoothing
cannyThresh = [0.12 0.30];     % edge thresholds (low, high) - tune
houghPeaksCount = 20;          % how many peaks to consider in Hough
houghFillGap = 30;             % join nearby Hough segments
houghMinLen = 40;              % minimum segment length
smoothingAlpha = 0.2;          % smoothing between frames (0..1)
lineThickness = 6;             % thickness of drawn lane line (pixels)

% Video setup
v = VideoReader(videoFile);
writer = VideoWriter(outputFile);
open(writer);

prevLeftFit = [];
prevRightFit = [];

figure('Name','Lane Detection','NumberTitle','off');

while hasFrame(v)
    frame = readFrame(v);
    [outFrame, prevLeftFit, prevRightFit] = processFrame(frame, prevLeftFit, prevRightFit, ...
        gaussSigma, cannyThresh, houghPeaksCount, houghFillGap, houghMinLen, smoothingAlpha, lineThickness);
    imshow(outFrame)
    title('Lane Detection'); drawnow
    writeVideo(writer, outFrame);
end

close(writer);
disp(['Output written to: ' outputFile]);

%%%%%%%%%%%%%%%%%%%%%
% Local functions
%%%%%%%%%%%%%%%%%%%%%
function [overlayImg, leftFitOut, rightFitOut] = processFrame(frame, prevLeftFit, prevRightFit, ...
        gaussSigma, cannyThresh, houghPeaksCount, houghFillGap, houghMinLen, smoothingAlpha, lineThickness)

    rows = size(frame,1);
    cols = size(frame,2);

    % 1) Preprocess
    gray = rgb2gray(frame);
    blurred = imgaussfilt(gray, gaussSigma);

    % 2) Edge detection
    edges = edge(blurred, 'Canny', cannyThresh);

    % 3) Region of interest (trapezoid)
    % coordinates relative to image size; adjust as needed for your camera
    bottom_left  = [round(cols*0.1), rows];
    top_left     = [round(cols*0.45), round(rows*0.6)];
    top_right    = [round(cols*0.55), round(rows*0.6)];
    bottom_right = [round(cols*0.9), rows];

    roiPolyX = [bottom_left(1), top_left(1), top_right(1), bottom_right(1)];
    roiPolyY = [bottom_left(2), top_left(2), top_right(2), bottom_right(2)];
    mask = poly2mask(roiPolyX, roiPolyY, rows, cols);
    roiEdges = edges & mask;

    % 4) Hough transform
    [H, theta, rho] = hough(roiEdges);
    peaks = houghpeaks(H, houghPeaksCount, 'Threshold', ceil(0.3*max(H(:))));
    lines = houghlines(roiEdges, theta, rho, peaks, 'FillGap', houghFillGap, 'MinLength', houghMinLen);

    % 5) Separate left and right lane points
    left_x = []; left_y = [];
    right_x = []; right_y = [];

    for k = 1:length(lines)
        p1 = lines(k).point1;
        p2 = lines(k).point2;
        dx = (p2(1) - p1(1));
        dy = (p2(2) - p1(2));
        slope = dy / (dx + eps);

        % Filter near-horizontal lines
        if abs(slope) < 0.3
            continue;
        end

        if slope < 0  % left lane (negative slope in image coords)
            left_x = [left_x; p1(1); p2(1)];
            left_y = [left_y; p1(2); p2(2)];
        else          % right lane
            right_x = [right_x; p1(1); p2(1)];
            right_y = [right_y; p1(2); p2(2)];
        end
    end

    leftFitOut = prevLeftFit;
    rightFitOut = prevRightFit;

    overlayImg = frame;

    % vertical range (where to draw lines)
    yBottom = rows;
    yTop = round(rows*0.6);  % same as ROI top

    % 6) Fit and draw left lane (linear fit x = m*y + b)
    if ~isempty(left_x)
        pLeft = polyfit(left_y, left_x, 1); % linear fit (degree=1)
        if ~isempty(prevLeftFit)
            pLeft = smoothingAlpha .* pLeft + (1 - smoothingAlpha) .* prevLeftFit;
        end
        leftFitOut = pLeft;
        xBottom = polyval(pLeft, yBottom);
        xTop    = polyval(pLeft, yTop);
        overlayImg = drawLineOnImage(overlayImg, [xBottom yBottom], [xTop yTop], [255 255 0], lineThickness);
    end

    % 7) Fit and draw right lane
    if ~isempty(right_x)
        pRight = polyfit(right_y, right_x, 1);
        if ~isempty(prevRightFit)
            pRight = smoothingAlpha .* pRight + (1 - smoothingAlpha) .* prevRightFit;
        end
        rightFitOut = pRight;
        xBottom = polyval(pRight, yBottom);
        xTop    = polyval(pRight, yTop);
        overlayImg = drawLineOnImage(overlayImg, [xBottom yBottom], [xTop yTop], [255 255 0], lineThickness);
    end

    % OPTIONAL: you could fill polygon between lanes (lane area) by sampling points,
    % creating an alpha-blended polygon, etc. (not included here)
end

function imgOut = drawLineOnImage(imgIn, pt1, pt2, color, thickness)
    % Draw thick line on RGB image by rasterizing between two points.
    % pt1, pt2 = [x y] in image coordinates.
    imgOut = imgIn;
    [rows, cols, ~] = size(imgIn);
    x1 = round(pt1(1)); y1 = round(pt1(2));
    x2 = round(pt2(1)); y2 = round(pt2(2));
    n = max(abs(x2-x1), abs(y2-y1));
    if n == 0
        return
    end
    xs = round(linspace(x1, x2, n));
    ys = round(linspace(y1, y2, n));
    half = floor(thickness/2);
    for i = 1:length(xs)
        x = xs(i); y = ys(i);
        for dx = -half:half
            for dy = -half:half
                xi = x + dx; yi = y + dy;
                if xi >= 1 && xi <= cols && yi >= 1 && yi <= rows
                    imgOut(yi, xi, :) = uint8(color);
                end
            end
        end
    end
end
