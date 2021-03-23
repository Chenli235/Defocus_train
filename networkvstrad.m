
filepath = 'F:/lightsheet_data/test/';
%levels = [8,,14,18,22,26,30,34,38,42,46];
levels = [8,11,14,17,20,23,26,29,32,35,38,41,44];
total_run = 10;img_size = 128;
num_images = length(dir([filepath,num2str(levels(1)),'/*.tiff']));
for i=1:13
    file{i} = dir([filepath,num2str(levels(i)),'/*.tiff']);
end
dist_focus = 0;
%all_score = zeros(200,11);
for num = 1:total_run
    disp(num)
    for i =1:42
        % read image
        %disp(i);
        for j=1:13
            img{j} = imread([file{j}(i).folder,'/',file{j}(i).name]);
        end
        if i == 0
          disp([num,i]);  
        end
        
        height = size(img{1},1);
        width = size(img{1},2);
        % crop into 124*124 size
        x = ceil((height - img_size-1)*rand());
        y = ceil((width - img_size-1)*rand());
%         x = ceil(height/2-64);
%         y = ceil(width/2-64);
        for j=1:13
            img{j} = img{j}(x:x+128-1,y:y+128-1);
        end
        %tic;
        % feed into image metric
        tic;
        score = measure_img(img,'LAPV');
        
%         if score(6)<500000
%             all_score(i,:) = score;
%         end%toc;
        %all_score(i,:) = score;
        [a,index] = max(score);
        all_score((num-1)*42+i,:) = index;
        dist_focus = dist_focus + abs(index-7);
    end
end
dist_focus/(total_run*42)

% VARS 2.0429 12.257
% LAPE 3.6476
% LAPV 3.1429 18.857
% VOLA 1.219  7.314
% BREN 1.4095 8.457
% TENV 1.0667 6.4002

% WAVS 3.0857
% WAVV 1.8857 11.314
% DCTS 0.9333 5.5998
% Network with cert 0.61194 3.6716
% Network without cert 0.85238 5.1143

%plot(all_score,'.','MarkerSize', 15);axis([1 200 1 11]);title('DCTS');line([1,200],[6,6],'Color','red','LineStyle','--');
%disp(dist_focus/200/10);
% load handel
% sound(y,Fs)
% x = [-5:5];
% sz_DCTS = zeros(1,13);
% for ii =1:total_run*42
%     sz_DCTS(all_score(ii)) = sz_DCTS(all_score(ii)) + 1;
% end
% x = ones(1,13);
% y = -6:6;
% all_sz = {sz_VARS,sz_LAPV,sz_LAPD,sz_WAVV,sz_SFIL,sz_TENV,sz_BREN,sz_DCTS};
% for i = 1:8
%     x = i*ones(1,13);
%     y = -6:6;
%     bubblechart(x,y,all_sz{i});hold on;
% end
% bubblelegend('Number of predication','Location','Northeastoutside','Style','telescopic','NumBubbles',3);
% set(gca,'xtick',[]);set(gca,'ytick',[]);


% LAPE 2.23
% LAPM 2.121
% LAPV 1.9645
% LAPD 1.7135
% VARS 1.3945
% BREN 0.811
% VOLA 0.824
% TENV 0.8015
% TENG 0.7755
% WAVS 1.627
% WAVV 0.951
% DCTS 0.832(r=40),0.789(r = 50),  0.7815(r=60), 0.7845(r=70), 0.746(r=80),
% 0.75(r=90), 0.716(r=100,time=0.117331), 0.748(r=110), 0.758(r=120)