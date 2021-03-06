function p = calcpesq(tgtname, refname, pesqexe)
%% CALCPESQ
%% PESQ based on ITU-T Recommendation P.862.2
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



% Run PESQ software. 
%----------------------------------------------------------------------

if exist('pesq_results.txt', 'var')
  delete('pesq_results.txt');
end

CMD = [pesqexe, ' +16000 ', refname, ' ', tgtname, ' > TMP'];
system(CMD);


% Extract the result.

%----------------------------------------------------------------------

fid = fopen('pesq_results.txt');

l = fgetl(fid);  %% first line is discarded
l = fgetl(fid);  %% second line contains the result

[a, l] = strtok(strtrim(l));  %% remove reference file name
[a, l] = strtok(strtrim(l));  %% remove target file name
[a, l] = strtok(strtrim(l));  %% remove target file name

p = str2num(strtok(strtrim(l)));

fclose(fid);

% delete('pesq_results.txt');
delete('TMP');

