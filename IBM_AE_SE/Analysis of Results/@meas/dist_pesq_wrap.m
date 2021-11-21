function [moslqo,rawmos]=dist_pesq_wrap(clean,degraded,fs,mode)

% narrow band analysis is the default
if nargin<4
    mode='nb';
end

% move to the directory where the EXE is
curr_path=mfilename('fullpath');
tmp=strfind(curr_path,'\');
orig_dir=cd(curr_path(1:tmp(end)-1));

% run the command
if strcmp(mode,'wb');
    [~,stdout]=system(['pesq +wb +' num2str(fs) ' "' clean '" "' degraded '"']);
else
    [~,stdout]=system(['pesq +' num2str(fs) ' "' clean '" "' degraded '"']);
end

% move back to the original directory
cd(orig_dir);

% parse the output log
indx=strfind(stdout,'=');
if isempty(indx)
    disp(' ');
    disp('! pesqwrap: pesq.exe failed!');
    disp('! For more details, read the following:');
    disp(' ');
    error(stdout);
end

% get the result from output log
res=str2num(stdout(indx+1:end));
moslqo=res(end);

% for wideband analysis - no RawMOS output
if strcmp(mode,'wb');
    rawmos='';
else
    rawmos=res(1);
end
