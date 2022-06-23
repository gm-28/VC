close all
clear

tp=0;
fp=0;
fn=0;
tn=0;
        
imgs = dir('images\*.jpg') ;         % folder of jpg images
masks = dir('masks\*.png') ;         % folder of png mask
N_imgs = length(imgs) ;              % total number of images
N_masks = length(masks) ;            % total number of masks

jac_v=zeros(N_imgs,1);               % array with jaccard values
imgs_name_vec=strings(N_imgs,1);     % array with images names

state = strings(N_imgs);
true_state = readtable('GT_Task2_Test.csv','Delimiter','comma');
true_state = table2array(true_state); 

for ni=1:N_imgs
    img_name=fullfile('images',imgs(ni).name);
    imgs_name_vec(ni)=convertCharsToStrings(imgs(ni).name);
    img = imread(img_name);
    mask_name=fullfile('masks',masks(ni).name);
    true_mask = imread(mask_name);
    
    bin = bwareafilt(true_mask, 1);
    stats = regionprops('table',bin,'Centroid','MajorAxisLength','MinorAxisLength','BoundingBox');
    w = stats.MinorAxisLength;
    h = stats.MajorAxisLength;
    X=stats.Centroid(1)-w/2;
    Y=stats.Centroid(2)-h/2;
    box = [X,Y,w,h];
    img_2 = imcrop(img, box);
   
    img_3 = im2double(imbinarize(im2gray(img_2),0.2));
    
    ybr = rgb2ycbcr(img_2);
    ybr = im2double(ybr);
    Cr = ybr(:,:,3);
    Cr2=(Cr.^2);
    Cr2=rescale(Cr2,0,255);
    
    Cb = ybr(:,:,2);
    CrCb = (Cr./Cb);
    rescale(CrCb,0,255);
    
    l = length( img_2(img_2~=0));
    num = (sum(Cr2(:))/l);
    den = (sum(CrCb(:))/l);
    n = 0.95*(num/den);
    
    Mouth_map = Cr2.*((Cr2-n*CrCb).^2);
    Mouth_map = rescale(Mouth_map,0,1);
    
    th = mean2(Mouth_map);
    Mouth_map1 = imbinarize(Mouth_map,10*th);
    
    bin = bwareafilt(Mouth_map1, 1);
    stats = regionprops('table',bin,'Centroid');
        
    while(isempty(stats))
        th = th*0.8;
        Mouth_map1 = imbinarize(Mouth_map,10*th);
        bin = bwareafilt(Mouth_map1, 1);
        stats = regionprops('table',bin,'Centroid');
    end
        
    X1=stats.Centroid(1)-(w*0.7)/2;
    Y1=stats.Centroid(2)-(h*0.3)/2;

    w1 = w*0.7;
    h1 = h*0.3;
    if((X1 + (w1)) > w)
        w1 = (w-X1);
    end

    if((Y1 + (h1)) > h)
        h1 = h-Y1;
    end
    box = [X1, Y1, w1 ,h1];

    squared=insertObjectAnnotation(img_2,'rectangle',box,'1','linewidth',5);
    croppedImage1 = imcrop(Mouth_map, box);
    croppedImage2 = imcrop(img_3, box);
    croppedImage3 = imcrop(img_2, box);
    gray = im2double(rgb2gray(croppedImage3));

    rg = croppedImage3(:,:,1)- croppedImage3(:,:,2);
    rg = double(rg);
    m = mean2(rg);
    rg = rg - 0.6*m;

    [k1,k2,k3]=size(croppedImage3);
    h=zeros(k1,k2);
    s_v=0;

    [rg_sorted,idx] = sort(rg(:), 'descend');
    idx(rg(:)==0) = [];
    [row, col] = ind2sub(size(rg), idx);
    indexes = [row col];

    [rg_sorted2,idx2] = sort(rg(:), 'ascend');
    idx2(rg(:)==0) = [];
    [row, col] = ind2sub(size(rg), idx2);
    indexes2 = [row col];

    max_mask = zeros(k1,k2);
    min_mask = zeros(k1,k2);
    k=1;
    y=0;
    z=0;
    while s_v < k1*k2/20
        if((z+1)*floor(k1*k2*0.1) < k1*k2/20)
            for x=z*floor(k1*k2*0.1)+1:(z+1)*floor(k1*k2*0.1) 
                max_mask(indexes(x,1),indexes(x,2)) = 1;
            end
        end
        z=z+1;
        f=length(max_mask(max_mask~=0));
        mouth = double(bwareafilt(logical(max_mask), 1));
        skin = max_mask-mouth;
        rg2 = rg.* skin;
        s_v=length(rg2(rg2~=0));

        if(s_v < k1*k2/20)
            for x=y*floor(k1*k2/15)+1:(y+1)*floor(k1*k2/15) 
                    min_mask(indexes2(x,1),indexes2(x,2)) = rg_sorted2(x);    
            end
            y=y+1;
            rg2 = rg2 + min_mask;
        end
        s_v=length(rg2(rg2~=0));
    end
    t = maxk(rg2(:),1);

    for i=1:k1
        for j=1:k2
            if(rg(i,j) <= t)
               h(i,j)=1;
            elseif (rg(i,j) > t)
               h(i,j)=0;
            end
        end
    end

    bin = bwareafilt(logical(h), 4);
    stats = regionprops('table',bin,'Centroid','Image');
    [s1,s2]=size(stats);
    center=[floor(k1/2), floor(k2/2)];
    dmin=1000;
    imin=0;

    for i=1:s1
        d=sqrt((stats.Centroid(i,1)-center(2))^2+(stats.Centroid(i,2)-center(1))^2);
        if(d<dmin) 
            dmin=d;
            imin=i;
        end
    end

    if(dmin<k2/4)
        l=stats.Image(imin);
        l=cell2mat(l);
        state(ni) = 'yawn';
    else
        state(ni) = 'no_yawn';
    end

    if(strcmp(state(ni),true_state(ni,2)))
        if(strcmp(state(ni),'yawn'))
            tp=tp+1;
        elseif(strcmp(state(ni),'no_yawn'))
            tn=tn+1;
        end
    else
        if(strcmp(state(ni),'yawn'))
            fp=fp+1;
        elseif(strcmp(state(ni),'no_yawn'))
            fn=fn+1;
        end           
    end
end

Recall = tp/(tp+fn);
Precision = tp/(tp+fp);
F1 = 2*Precision*Recall/(Precision+Recall);

writematrix(Recall,'Task_2__3_Test_Data.xlsx','Sheet',1,'Range','H9');
writematrix(Precision,'Task_2__3_Test_Data.xlsx','Sheet',1,'Range','H10');
writematrix(F1,'Task_2__3_Test_Data.xlsx','Sheet',1,'Range','H11');
writematrix(tp,'Task_2__3_Test_Data.xlsx','Sheet',1,'Range','G5');
writematrix(fp,'Task_2__3_Test_Data.xlsx','Sheet',1,'Range','H5');
writematrix(fn,'Task_2__3_Test_Data.xlsx','Sheet',1,'Range','G6');
writematrix(tn,'Task_2__3_Test_Data.xlsx','Sheet',1,'Range','H6');

