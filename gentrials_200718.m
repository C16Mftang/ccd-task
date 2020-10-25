%% Basic set-up
close all;
clearvars;
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1); 
% ignore warning msg output; won't be a problem when presented on experimental monitor
Screen('Preference','VisualDebugLevel', 0);
screens = Screen('Screens');
% automatically displays to highest numbered screen, e.g. external screen
% if there is one present
screenNumber = max(screens);
white = [255,255,255];sca

black = [0,0,0];
% recalculate mean screen luminance of white dots on black for grey (eventually)
grey = white/2;

%% This script creates ntotaltrials split into .mov files of ten trials each

ntotaltrials = 10;
ntrials = 10;
nfiles = ntotaltrials/ntrials;
% save video file in dimensions dim(width,height)
% except this is not working and the video file is exactly the size of the
% monitor... 
% dim = [36 64];
% set up parameters
coh = 1;
incoh = .15;
dirs = [0, 90, 180, 270];
dsize = 9; % pixels
dspeed = 45;
ndots = 150; % originally 460
catch_dur = 4;
n_catch_trials = round(ntrials/2);
n_change_trials = ntrials - n_catch_trials;
% change will take place at a variable time (500 - 3500 ms)
change_time_min = 0.5;
change_time_max = 3.5;
change_time_detect = 0.6;
grey_dur = 4;

% set up dots
dots.nDots = ndots;
dots.color = white;
dots.speed = dspeed/3;
dots.size = dsize;
dots.coherence = coh;
dots.direction = dirs(1);
dots.center = [0,0];
dots.apertureSize = [42, 42];

%% Begin loop to create nfiles with ntrials each

for fileidx = 1:nfiles
    % you may want to split this up; depending on your set-up you may not be able to use your screen as the script runs
    
    % set up trial sequence for this mov file
    seq = zeros(1,ntrials);
    seq(randperm(ntrials,n_catch_trials)) = 1; % 1 means change, 0 means no change
    trialdir = zeros(1,ntrials);
    trialcoh = zeros(1,ntrials);
    trialcoh(randperm(ntrials,n_catch_trials)) = 1; % 1 means start at coh, 0 means incoh
    changetimes = zeros(1,ntrials);
    trialend = zeros(1,ntrials);
    for ii = 1:length(trialdir) % 1:ntrials
        diridx = randi([1,4]);
        trialdir(ii) = dirs(diridx); % one of the four directions
        if seq(ii)~=0 % seq=1 means change, 0 means no change
            % Randomly choose when the change occurs during change trials
            changetimes(ii) = randi([change_time_min*1000, change_time_max*1000])/1000;
            % Trial ends 0.6s after the change occurs and next trial begins
            % trialend(ii) = changetimes(ii) + change_time_detect; 
            % to keep each trial at the same length (frames), remove this for now
            trialend(ii) = catch_dur;
        else
            trialend(ii) = catch_dur;
        end
    end

    % calculate total experiment duration
    duration = sum(trialend);

    %% Make stim!
    % open window and color the background black
    % changed the "rect" parameter here to open only a small (non-full screen) window
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, black, [0 0 320 180]);
    display.frameRate = FrameRate(window);
    % not necessary: dots.lifetime = display.frameRate*duration;
    display.resolution = zeros(1,2);
    [display.resolution(1), display.resolution(2)] = Screen('WindowSize',window);
    display.width = 30; % of monitor itself; cm (can change)
    display.dist = 50; % of monitor itself; cm (can change)
    saveVideo = true;
    createDriftingDots(coh,incoh,window,display,dots,grey,grey_dur,seq,trialdir,trialcoh,trialend,changetimes,saveVideo,fileidx);

% Saves the movie file and its corresponding info, i.e. trial sequence,
% trial type, coherence change time (if any)

%% Other considerations
% May eventually want to temporally extrapolate s.t. each "frame" is 1ms.

end
Screen('CloseAll');
