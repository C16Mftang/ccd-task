function createDriftingDots(coh,incoh,window,display,dots,grey,grey_dur,seq,trialdir,trialcoh,trialend,changetimes,saveVideo,fileidx)
% createDriftingDots
% input: window pointer, info about the display, dots structure, 
% total duration, sequence of trial types, change time matrix
% output: none; option to save video of stim
if saveVideo
    % add functionality to save video
    movieFile = sprintf('/Users/tangmufeng/Desktop/UChicago/Courses/MacleanLab_research/tfrecord_data_processing/fullres/stim_%i.mov',fileidx);
    movie = Screen('CreateMovie', window, movieFile);
    % in addition, save output txt file about the parameters of this movie
    % i.e. seq (0s and 1s denoting whether there is a change or not
    % changetimes, trialdir, trialcoh (starting coherence type), etc.
    fileName = sprintf('/Users/tangmufeng/Desktop/UChicago/Courses/MacleanLab_research/tfrecord_data_processing/fullres/trialinfo/stim_%i_info.mat',fileidx);
    % Create a table with the data and variable names
    framerate = display.frameRate;
    save(fileName,'seq','trialdir','trialend','grey_dur','trialcoh','changetimes','framerate');
    % save text file about all the variables
    %fileID = fopen('/Users/yuqingzhu/stim/varinfo.txt','w');
    %fprintf(fileID,'Variable info:');
    %fprintf(fileID,'seq tells whether each trial is a change (1) or no-change/catch (0) trial.');
    %fprintf(fileID,'trialdir tells which global direction out of 4 directions (0, 90, 180, 270) the dots are drifting in each trial.');
    %fprintf(fileID,'trialend gives the time (in s) at which each stimulus trial ends.');
    %fprintf(fileID,'trials which have no change end at 4s.');
    %fprintf(fileID,'there is a 3s gray screen between each trial, which grey_dur tells you.');
    %fprintf(fileID,'trialcoh tells which coherence value occurs first in each trial: 100 (1) or 15 (0) percent coherent.');
    %fprintf(fileID,'If it is a no-change trial, that is the only coherence value for the whole trial. If it is a change trial, the coherence value changes at the time specified in changetimes.');
    %fprintf(fileID,'changetimes tells the time (in seconds) at which the coherence change occurs. if the value is 0, no change occurs in that trial.');
    %fprintf(fileID,'a trial which changes coherence will do so at a variable time between 0.5 and 3.5s.');
    %fprintf(fileID,'the trial ends 0.6s after the coherence change.');
    %fprintf(fileID,'in order to find the frame number corresponding to the times in the variables above, do round(framerate*time).');
    %fclose(fileID);
end
%
% Animates a field of moving dots based on parameters defined in the 'dots'
% structure over a period of seconds defined by 'duration'.
%
% The 'dots' structure must have the following parameters:
%
%   nDots            Number of dots in the field
%   speed            Speed of the dots (degrees/second)
%   direction        Direction 0-360 clockwise from upward
%   apertureSize     [x,y] size of elliptical aperture (degrees)
%   center           [x,y] Center of the aperture (degrees)
%   color            Color of the dot field [r,g,b] from 0-255
%   size             Size of the dots (in pixels)
%   coherence        Coherence from 0 (incoherent) to 1 (coherent)
%
% 'dots' can be an array of structures so that multiple fields of dots can
% be shown at the same time.  The order that the dots are drawn is
% scrambled across fields so that one field won't occlude another.
%
% The 'display' structure requires the fields:
%    width           Width of screen (cm)
%    dist            Distance from screen (cm)
% And can also use the fields:
%    skipChecks      If 1, turns off timing checks and verbosity (default 0)
%    fixation        Information about fixation (see 'insertFixation.m')
%    screenNum       screen number       
%    bkColor         background color (default is [0,0,0])
%    windowPtr       window pointer, set by 'OpenWindow'
%    frameRate       frame rate, set by 'OpenWindow'
%    resolution      pixel resolution, set by 'OpenWindow'

% 3/23/09 Written by G.M. Boynton at the University of Washington
% modified 7/1/20 by Yuqing Zhu

nDots = dots.nDots;

%Set up the color and size vectors
colors = 255*ones(3,nDots);
sizes = dots.size*ones(1,nDots);

% calculate left, right, top, and bottom of each aperture (in degrees)
l = dots.center(1)-dots.apertureSize(1)/2;
r = dots.center(1)+dots.apertureSize(1)/2;
b = dots.center(2)-dots.apertureSize(2)/2;
t = dots.center(2)+dots.apertureSize(2)/2;

%% Set up experiment!

for ii = 1:length(seq)
    % for each trial in seq, initialize dots' starting positions
    dots.x = (rand(1,dots.nDots)-.5)*dots.apertureSize(1) + dots.center(1);
    dots.y = (rand(1,dots.nDots)-.5)*dots.apertureSize(2) + dots.center(2);
    % if no change, continue for 4s at correct coherence level
    if seq(ii)==0
        if trialcoh(ii)==1
            coherence = coh;
        else
            coherence = incoh;
        end
        % define appropriate coherence direction vector
        direction = rand(1,dots.nDots)*360;
        nCoherent = ceil(coherence*dots.nDots);  %Start w/ all random directions
        direction(1:nCoherent) = trialdir(ii);  %Set the 'coherent' directions
        %Calculate dx and dy vectors in real-world coordinates
        dots.dx = dots.speed*sin(direction*pi/180)/display.frameRate;
        dots.dy = -dots.speed*cos(direction*pi/180)/display.frameRate;
        % loop through frames
        nFrames = round(display.frameRate*trialend(ii));
        for frameNum=1:nFrames
            %Update the dot position's real-world coordinates
            dots.x = dots.x + dots.dx;
            dots.y = dots.y + dots.dy;

            %Move the dots that are outside the aperture back one aperture width.
            dots.x(dots.x<l) = dots.x(dots.x<l) + dots.apertureSize(1);
            dots.x(dots.x>r) = dots.x(dots.x>r) - dots.apertureSize(1);
            dots.y(dots.y<b) = dots.y(dots.y<b) + dots.apertureSize(2);
            dots.y(dots.y>t) = dots.y(dots.y>t) - dots.apertureSize(2);
            
            %Calculate the screen positions from the real-world coordinates
            pixpos.x = angle2pix(display,dots.x)+ display.resolution(1)/2;
            pixpos.y = angle2pix(display,dots.y)+ display.resolution(2)/2;

            % draw all dots
            Screen('DrawDots',window,[pixpos.x;pixpos.y],sizes,colors,[0,0],1);
            % save to movie file
            Screen('AddFrameToMovie',window);
            % flip to screen
            Screen('Flip', window);
        end
  
    else % if change, continue at specified coherence level until changetimes(ii)
        if trialcoh(ii)==1
            coherence = coh;
        else
            coherence = incoh;
        end
        % define appropriate coherence direction vector
        direction = rand(1,dots.nDots)*360;
        nCoherent = ceil(coherence*dots.nDots);  %Start w/ all random directions
        direction(1:nCoherent) = trialdir(ii);  %Set the 'coherent' directions
        %Calculate dx and dy vectors in real-world coordinates
        dots.dx = dots.speed*sin(direction*pi/180)/display.frameRate;
        dots.dy = -dots.speed*cos(direction*pi/180)/display.frameRate;
        % loop through frames
        % tracking position; at change time those positions seed new
        % coherence movement
        nFrames = round(changetimes(ii)*display.frameRate);
        for frameNum = 1:nFrames
            %Update the dot position's real-world coordinates
            dots.x = dots.x + dots.dx;
            dots.y = dots.y + dots.dy;

            %Move the dots that are outside the aperture back one aperture width.
            dots.x(dots.x<l) = dots.x(dots.x<l) + dots.apertureSize(1);
            dots.x(dots.x>r) = dots.x(dots.x>r) - dots.apertureSize(1);
            dots.y(dots.y<b) = dots.y(dots.y<b) + dots.apertureSize(2);
            dots.y(dots.y>t) = dots.y(dots.y>t) - dots.apertureSize(2);
            
            %Calculate the screen positions from the real-world coordinates
            pixpos.x = angle2pix(display,dots.x)+ display.resolution(1)/2;
            pixpos.y = angle2pix(display,dots.y)+ display.resolution(2)/2;

            % draw all dots
            Screen('DrawDots',window,[pixpos.x;pixpos.y],sizes,colors,[0,0],1);
            % save to movie file
            Screen('AddFrameToMovie',window);
            % flip to screen
            Screen('Flip', window);
        end
        
        if trialcoh(ii)==1 % now change coherence
            coherence = incoh;
        else
            coherence = coh;
        end
        % define appropriate coherence direction vector
        direction = rand(1,dots.nDots)*360;
        nCoherent = ceil(coherence*dots.nDots);  %Start w/ all random directions
        direction(1:nCoherent) = trialdir(ii);  %Set the 'coherent' directions
        %Calculate dx and dy vectors in real-world coordinates
        dots.dx = dots.speed*sin(direction*pi/180)/display.frameRate;
        dots.dy = -dots.speed*cos(direction*pi/180)/display.frameRate;
        
        % specify number of remaining frames
        remainingFrames = round((trialend(ii)-changetimes(ii))*display.frameRate);
        for frameNum = 1:remainingFrames
            %Update the dot position's real-world coordinates based on new
            %coherent motion
            dots.x = dots.x + dots.dx;
            dots.y = dots.y + dots.dy;

            %Move the dots that are outside the aperture back one aperture width.
            dots.x(dots.x<l) = dots.x(dots.x<l) + dots.apertureSize(1);
            dots.x(dots.x>r) = dots.x(dots.x>r) - dots.apertureSize(1);
            dots.y(dots.y<b) = dots.y(dots.y<b) + dots.apertureSize(2);
            dots.y(dots.y>t) = dots.y(dots.y>t) - dots.apertureSize(2);
            
            %Calculate the screen positions from the real-world coordinates
            pixpos.x = angle2pix(display,dots.x)+ display.resolution(1)/2;
            pixpos.y = angle2pix(display,dots.y)+ display.resolution(2)/2;

            % draw all dots
            Screen('DrawDots',window,[pixpos.x;pixpos.y],sizes,colors,[0,0],1);
            % save to movie file
            Screen('AddFrameToMovie',window);
            % flip to screen
            Screen('Flip', window);
        end 
    end
    % finally grey screen for 3s, seeded with the proper positions
    greyFrames = round(grey_dur*display.frameRate);
    for frameNum = 1:greyFrames
        Screen('FillRect',window,0.3);
        Screen('AddFrameToMovie',window);
        Screen('Flip',window);
        pause(1/display.frameRate);
    end
    %Screen('FillRect',window,0.3);
    %Screen('Flip', window);
    %pause(grey_dur);
    Screen('FillRect',window,0); % remake black background
    Screen('Flip',window);
end 
if saveVideo
    % finalize and save video
    Screen('FinalizeMovie',movie);
end 
Screen('CloseAll');

