% img_original = imread('../913375_-0.0124.tiff');
% img_displaced_1 = imread('../913375_-0.0324_1.tiff');
% img_displaced_2 = imread('../913375_-0.0324_2.tiff');
% img_correct_1 = imread('../913375_0.00117_aftercorrection.tiff');
% img_original_1 = img_original(128*4+1:128*12,128*4+1:128*12);
% img_displaced = img_displaced_1(128*4+1:128*12,128*4+1:128*12);
% img_correct = img_correct_1(128*4+1:128*12,128*4+1:128*12);
% imwrite(img_original_1,'original_lung2.tiff');
% imwrite(img_displaced,'displaced_lung2.tiff');
% imwrite(img_correct,'corrected_lung2.tiff');
% img(:,:,1) = img_displaced_1;
% img(:,:,2) = img_displaced_2;
% [level,pred,cert] = imgstack_pred(img);
% level
% pred
% cert
% border_patches = zeros(8*8,128,128,3);
%pred1 = pred';
% img = imread('original_lung2.tiff');
% img1 = img(512-150:512+150,512-150:512+150);
% imwrite(img1,'lung_2_original.tiff');
% 
% img = imread('displaced_lung2.tiff');
% img1 = img(512-150:512+150,512-150:512+150);
% imwrite(img1,'lung_2_displaced.tiff');
% 
% img = imread('corrected_lung2.tiff');
% img1 = img(512-150:512+150,512-150:512+150);
% imwrite(img1,'lung_2_corrected.tiff');



% imwrite(img1,'2_corrected.tiff')
% imwrite(img2,'2_corrected2.tiff')
% clear;
% filepath = 'F:/lightsheet_data/brain/train/';
%levels = [8,11,14,18,22,26,30,34,38,42,46];
filepath = 'F:/lightsheet_data/test/';
levels = [8,11,14,17,20,23,26,29,32,35,38,41,44];
total_run = 10;img_size = 128;
num_images = length(dir([filepath,num2str(levels(1)),'/*.tiff']));
for i=1:13
    file{i} = dir([filepath,num2str(levels(i)),'/*.tiff']);
end
file{14} = dir([filepath,num2str(47),'/*.tiff']);
dist_focus = 0;
true_patch = 0;
num_predicted = 0;
for num = 1:total_run
    disp(num)
    for i = 1:42
        
        choose_level = ceil(13*rand);
        %disp(choose_level);
        img1 = imread([file{choose_level}(i).folder,'/',file{choose_level}(i).name]);
        img2 = imread([file{choose_level+1}(i).folder,'/',file{choose_level+1}(i).name]);
        actual_level = choose_level;
        height = size(img1,1);
        width = size(img1,2);
        % crop into 128*128 size 128*3 128*
        %x = ceil((height - img_size-1)*rand());
        %y = ceil((width - img_size-1)*rand());
        
        x = ceil((height - img_size-1)*rand());
        y = ceil((width - img_size-1)*rand());
        img1 = img1(x:x+128-1,y:y+128-1);
        img2 = img2(x:x+128-1,y:y+128-1);
        img(:,:,1) = img1;img(:,:,2)=img2;
        % prediction defocus level
        num_patches = 0;cert_weight = 0;cert_all = 0;
%         for index_i = 1:3
%             for index_j = 1:3
%                 img(:,:,1) = img1((index_i-1)*128+1:(index_i-1)*128+128,(index_j-1)*128+1:(index_j-1)*128+128);
%                 img(:,:,2) = img2((index_i-1)*128+1:(index_i-1)*128+128,(index_j-1)*128+1:(index_j-1)*128+128);
%                 res = py.pred_img_6nm.pred_onepatch(img);
%                 pred = single(res{1});
%                 cert = single(res{2});
%                 if cert < 0.35
%                     continue;
%                 end
%                 num_patches = num_patches + 1;
%                 cert_weight = cert_weight  + pred;
%                 cert_all = cert_all + cert;
%             end
%         end
%         if num_patches == 0
%             pred_defocus = 0;
%             num_predicted = num_predicted + 1;
%             dist_focus = dist_focus + abs(pred_defocus-(actual_level-1));
%         else
%             pred_defocus = cert_weight/num_patches;
%             num_predicted = num_predicted + 1;
%             dist_focus = dist_focus + abs(pred_defocus-(actual_level-1));
%         end
%         if num_patches >= 4
%             pred_defocus = cert_weight/num_patches;
%             num_predicted = num_predicted + 1;
%             dist_focus = dist_focus + abs(pred_defocus-(actual_level-1));
%         end
        res = py.pred_img_6nm.pred_onepatch(img);
        pred = single(res{1});
        cert = single(res{2});
        dist_focus = dist_focus + abs(pred-(actual_level-1));
        num_predicted = num_predicted + 1;

        
        %true_patch = true_patch+1;
    end
end
dist_focus/(num_predicted)
num_predicted
% img1 = imread('F:/defocus_dataset/new_1_20_2021/1_16_2021_brain/561nm/450541_083821_228976_10.tiff');
% img2 = imread('F:/defocus_dataset/new_1_20_2021/1_16_2021_brain/561nm/450541_083821_228976_18.tiff');
% img3 = imread('450541_083821_228976_26.tiff');
% img4 = imread('F:/defocus_dataset/new_1_20_2021/1_16_2021_brain/561nm/450541_083821_228976_34.tiff');
% img5 = imread('F:/defocus_dataset/new_1_20_2021/1_16_2021_brain/561nm/450541_083821_228976_42.tiff');
% 
% img11 = img1(200:1500,1000:1900);
% img22 = img2(200:1500,1000:1900);
% img33 = img3(200:1500,1000:1900);
% img44 = img4(200:1500,1000:1900);
% img55 = img5(200:1500,1000:1900);
% 
% imshow(img33,[0,500])
% 
% imwrite(img11,'450541_083821_228976_10_small.tiff');
% imwrite(img22,'450541_083821_228976_18_small.tiff');
% imwrite(img33,'450541_083821_228976_26_small.tiff');
% imwrite(img44,'450541_083821_228976_34_small.tiff');
% imwrite(img55,'450541_083821_228976_42_small.tiff');

% img = imread('800280_-0.0396_aftercorrection.tiff');
% img1 = img(1010:1310,750:1050);
% imshow(img1,[0,1000]);
% imwrite(img1,'1_corrected_2.tiff')

% img1 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_14.tiff');
% img2 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_17.tiff');
% img3 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_20.tiff');
% img4 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_23.tiff');
% img5 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_26.tiff');
% img6 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_29.tiff');
% img7 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_32.tiff');
% img8 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_35.tiff');
% img9 = imread('F:/defocus_dataset/new_1_20_2021/1_17_2021_brain/561nm/875942_550156_622475_38.tiff');
% img_1 = imread('lung_2_original.tiff');
% img_2 = imread('lung_2_displaced.tiff');
% img_3 = imread('lung_2_corrected.tiff');
% diag1 = diag(img_1);diag2 = diag(img_2);diag3 = diag(img_3);
% x=1:301;
% plot(x,diag1,x,diag2,x,diag3,'linewidth',2);
% %legend('Ground Truth','Displaced','Corrected','Fontsize',50);
% axis off;
% figure(2);
%legend('Ground Truth','Displaced','Corrected','Fontsize',30);
filepath = 'F:/lightsheet_data/test/';
levels = [8,11,14,17,20,23,26,29,32,35,38,41,44];
total_run = 10;img_size = 128;
num_images = length(dir([filepath,num2str(levels(1)),'/*.tiff']));
for i=1:13
    file{i} = dir([filepath,num2str(levels(i)),'/*.tiff']);
end
file{14} = dir([filepath,num2str(47),'/*.tiff']);
dist_focus = 0;
true_patch = 0;
num_predicted = 0;
for num = 1:total_run
    disp(num)
    for i = 1:42
        
        choose_level = ceil(13*rand);
        %disp(choose_level);
        img1 = imread([file{choose_level}(i).folder,'/',file{choose_level}(i).name]);
        img2 = imread([file{choose_level+1}(i).folder,'/',file{choose_level+1}(i).name]);
        actual_level = choose_level;
        height = size(img1,1);
        width = size(img1,2);
        % crop into 128*128 size 128*3 128*
        %x = ceil((height - img_size-1)*rand());
        %y = ceil((width - img_size-1)*rand());
        
        x = ceil((height - img_size-1)*rand());
        y = ceil((width - img_size-1)*rand());
        img1 = img1(x:x+128-1,y:y+128-1);
        img2 = img2(x:x+128-1,y:y+128-1);
        img(:,:,1) = img1;img(:,:,2)=img2;
        % prediction defocus level
        num_patches = 0;cert_weight = 0;cert_all = 0;
        res = py.pred_img_6nm.pred_onepatch(img);
        pred = single(res{1});
        cert = single(res{2});
        if cert > 0.35
            num_predicted = num_predicted + 1;
            dist_focus = dist_focus + abs(pred-(actual_level-1));
        end
    end
end
dist_focus/(num_predicted)
num_predicted