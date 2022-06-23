close all
clear

tp=0;
fn=0;
fp=0;

imgs = dir('images\*.jpg') ;         % folder of jpg images
masks = dir('masks\*.png') ;         % folder of png mask
N_imgs = length(imgs) ;              % total number of images
N_masks = length(masks) ;            % total number of masks

jac_v=zeros(N_imgs,1);               % array with jaccard values
imgs_name_vec=strings(N_imgs,1);     % array with images names

for ni=1:N_imgs
    img_name=fullfile('images',imgs(ni).name);
    imgs_name_vec(ni)=convertCharsToStrings(imgs(ni).name);
    img = imread(img_name);
    
    [k1,k2,k3]=size(img);
    msk=zeros(k1,k2);
    
    ybr=rgb2ycbcr(img);
    y=ybr(:,:,1);
    y_mean=mean(y,'all');
    y_max=max(y,[],'all');
         
    for i=1:k1
        for j=1:k2
            if(ybr(i,j,1)<100)
                ybr(i,j,1)= ybr(i,j,1) + 50;
            end
        end
    end
    
    img2=ycbcr2rgb(cat(3,ybr(:,:,1),ybr(:,:,2),ybr(:,:,3)));

  	for i=1:k1
        for j=1:k2
            y=ybr(i,j,1);
            cb=ybr(i,j,2);
            cr=ybr(i,j,3);
            
            r=img2(i,j,1);
            g=img2(i,j,2);
            b=img2(i,j,3);
            m=max([r g b]);
            n=min([r g b]);
            
            if  ((r>95)&&(g>40)&&(b>20)&&((m-n)>15)&&(abs(r-g)>15)&&(r>g)&&(r>b))&&((cb>=100)&& cb<=125 && cr>=135 && cr<=170 && y>=60 && y<=255 )
                msk(i,j)=1;
            else
                msk(i,j)=0;
            end
        end
    end
    
    gray = im2double(rgb2gray(img));

    se=strel('square',2);
    msk=imerode(msk,se);
    msk=imfill(msk,'holes');
    se=strel('square',4);
    msk=imopen(msk,se);
    msk=imdilate(msk,se);    
    fil=logical(msk);
    bin = bwareafilt(fil, 1);
   
    squared=img;
    stats = regionprops('table',bin,'Centroid','MajorAxisLength','MinorAxisLength');
    
    MajorAxis=stats.MajorAxisLength;
    MinorAxis=stats.MinorAxisLength;
    
    Y1=stats.Centroid(2)-MajorAxis/2; %Fixa altura do centroide antes de alterar altura da box para subir apenas a parte de baixo, evita shift
    if( (Y1 + MajorAxis)> 0.88*k1 || (MajorAxis > 0.75*k1)) 
           MajorAxis = 0.8*MajorAxis;
    end
    if(MinorAxis>0.4*k2)
           MinorAxis = 0.6*MinorAxis;
    end
    X1=stats.Centroid(1)-MinorAxis/2; % Shift horizontal caso exista alterações à largura da box
    box = [X1, Y1, MinorAxis, MajorAxis];
   
    squared=insertObjectAnnotation(squared,'rectangle',box,'1','linewidth',5);
   
    mask_name = fullfile('masks',masks(ni).name);
    true_mask = imread(mask_name);    

    n_mask = zeros(k1,k2);
    n_mask = insertShape(n_mask,'FilledRectangle',box,'Color','w');
    n_mask = imbinarize(rgb2gray(n_mask), 0);
    
    if(ni<10)
        FileName = sprintf('mask0%d.png', ni);
    else
        FileName = sprintf('mask%d.png', ni);
    end
    FilePath = fullfile('new_masks', FileName);
    imwrite(n_mask,FilePath);
    jac = jaccard(n_mask,true_mask);
      

    if jac>=0.5
        tp=tp+1;
    elseif jac<0.5
        fn=fn+1;
        fp=fp+1;
    end  

    jac_v(ni)=jac;
end

Recall = tp/(tp+fn);
Precision = tp/(tp+fp);
F1 = 2*Precision*Recall/(Precision+Recall);

writematrix(imgs_name_vec,'Task_1_Test_Data.xlsx','Sheet',1,'Range','B3');
writematrix(jac_v,'Task_1_Test_Data.xlsx','Sheet',1,'Range','C3');
writematrix(Recall,'Task_1_Test_Data.xlsx','Sheet',1,'Range','H2');
writematrix(Precision,'Task_1_Test_Data.xlsx','Sheet',1,'Range','H3');
writematrix(F1,'Task_1_Test_Data.xlsx','Sheet',1,'Range','H4');
writematrix(tp,'Task_1_Test_Data.xlsx','Sheet',1,'Range','H5');
writematrix(fp,'Task_1_Test_Data.xlsx','Sheet',1,'Range','H6');
writematrix(fn,'Task_1_Test_Data.xlsx','Sheet',1,'Range','H7');
