function score = measure_img(img,method)
    score = [];
    method = upper(method);
    if method == 'LAPE' % Energy of Laplacian
        for i = 1:size(img,2)
            score = [score,LAPE(img{i})];
        end
    elseif method == 'LAPM' % Modified Laplacian
        for i = 1:size(img,2)
            score = [score,LAPM(img{i})];
        end
    elseif method == 'LAPV' % Variance of laplacian (Pech2000)
        for i = 1:size(img,2)
            score = [score,LAPV(img{i})];
        end
    elseif method == 'LAPD' % Diagonal laplacian (Thelen2009)
        for i = 1:size(img,2)
            score = [score,LAPD(img{i})];
        end
    elseif method == 'VARS'
        for i = 1:size(img,2)
            score = [score,var(single(img{i}(:)))];
        end
    elseif method == 'BREN' % Brenner's
        for i = 1:size(img,2)
            score = [score,BREN(img{i})];
        end     
    elseif method == 'VOLA' % Vollath's correlation (Santos97)
        for i = 1:size(img,2)
            score = [score,VOLA(img{i})];
        end
    elseif method == 'TENV' % Tenengrad variance (Pech2000)
        for i = 1:size(img,2)
            score = [score,TENV(img{i})];
        end
    elseif method == 'TENG' % Tenengrad (Krotkov86)
        for i = 1:size(img,2)
            score = [score,TENG(img{i})];
        end
    elseif method == 'WAVS' % %Sum of Wavelet coeffs (Yang2003)
        for i = 1:size(img,2)
            score = [score,WAVS(img{i})];
        end
    elseif method == 'WAVV' %Variance of  Wav...(Yang2003)
        for i = 1:size(img,2)
            score = [score,WAVV(img{i})];
        end
    elseif method == 'DCTS' %Variance of  Wav...(Yang2003)
        for i = 1:size(img,2)
            
            score = [score,DCTS(img{i})];
            
        end
    elseif method == 'HISE'
        for i = 1:size(img,2)
            
            score = [score,HISE(img{i})];
            
        end
    elseif method == 'SFIL' % Streerable filters
        for i = 1:size(img,2)
            
            score = [score,SFIL(img{i})];
            
        end
    else
        score = 0;
    end

    function fm = BREN(img)
        [M,N] = size(img);
        DH = zeros(M,N);
        DV = zeros(N,N);
        DV(1:M-2,:) = img(3:end,:) - img(1:end-2,:);
        DH(:,1:N-2) = img(:,3:end) - img(:,1:end-2);
        FM = max(DH,DV);
        FM = FM.^2;
        fm = mean2(FM);
    end
    function fm = LAPE(img)
        LAP = fspecial('laplacian');
        FM = imfilter(img, LAP, 'replicate', 'conv');
        fm = mean2(FM.^2);
    end
    function fm = LAPM(img)
        M = [-1 2 -1];        
        Lx = imfilter(img, M, 'replicate', 'conv');
        Ly = imfilter(img, M', 'replicate', 'conv');
        FM = abs(Lx) + abs(Ly);
        fm = mean2(FM);
    end
    function fm = LAPV(img)
        LAP = fspecial('laplacian');
        ILAP = imfilter(img, LAP, 'replicate', 'conv');
        fm = std2(ILAP)^2;
    end
    function fm = LAPD(img)
        M1 = [-1 2 -1];
        M2 = [0 0 -1;0 2 0;-1 0 0]/sqrt(2);
        M3 = [-1 0 0;0 2 0;0 0 -1]/sqrt(2);
        F1 = imfilter(img, M1, 'replicate', 'conv');
        F2 = imfilter(img, M2, 'replicate', 'conv');
        F3 = imfilter(img, M3, 'replicate', 'conv');
        F4 = imfilter(img, M1', 'replicate', 'conv');
        FM = abs(F1) + abs(F2) + abs(F3) + abs(F4);
        fm = mean2(FM);
    end
    function fm = VOLA(img)
        Image = double(img);
        I1 = Image; I1(1:end-1,:) = Image(2:end,:);
        I2 = Image; I2(1:end-2,:) = Image(3:end,:);
        Image = Image.*(I1-I2);
        fm = mean2(Image);
    end
    function fm = TENV(img)
        Sx = fspecial('sobel');
        Gx = imfilter(double(img), Sx, 'replicate', 'conv');
        Gy = imfilter(double(img), Sx', 'replicate', 'conv');
        G = Gx.^2 + Gy.^2;
        fm = std2(G)^2;
    end
    function fm = TENG(img)
        Sx = fspecial('sobel');
        Gx = imfilter(double(img), Sx, 'replicate', 'conv');
        Gy = imfilter(double(img), Sx', 'replicate', 'conv');
        FM = Gx.^2 + Gy.^2;
        fm = mean2(FM);
    end
    function fm = WAVS(img)
        [C,S] = wavedec2(img, 1, 'db6');
        H = wrcoef2('h', C, S, 'db6', 1);   
        V = wrcoef2('v', C, S, 'db6', 1);   
        D = wrcoef2('d', C, S, 'db6', 1);   
        FM = abs(H) + abs(V) + abs(D);
        fm = mean2(FM);
    end
    function fm = WAVV(img)
        [C,S] = wavedec2(img, 1, 'db6');
        H = abs(wrcoef2('h', C, S, 'db6', 1));
        V = abs(wrcoef2('v', C, S, 'db6', 1));
        D = abs(wrcoef2('d', C, S, 'db6', 1));
        fm = std2(H)^2+std2(V)+std2(D);
    end
    function fm = DCTS(img)
        r=100;
        img_filt = imgaussfilt(img);
        img_DCT = dct2(img_filt);
        L2 = norm(img_DCT);
        value = 0;
        for i=1:r
            for j=1:r
                if i^2+j^2<r^2
                    val = img_DCT(i,j)/L2;
                    value = value + abs(val)*abslog2(val);
                else
                    continue;
                end
            end
        end
        fm = value*-2/(r^2); 
    end
    function fm = HISE(img)
        fm = entropy(img);
    end
    function fm = SFIL(img)
        Image = img;
        WSize = 15;
        N = floor(WSize/2);
        sig = N/2.5;
        [x,y] = meshgrid(-N:N, -N:N);
        G = exp(-(x.^2+y.^2)/(2*sig^2))/(2*pi*sig);
        Gx = -x.*G/(sig^2);Gx = Gx/sum(Gx(:));
        Gy = -y.*G/(sig^2);Gy = Gy/sum(Gy(:));
        R(:,:,1) = imfilter(double(Image), Gx, 'conv', 'replicate');
        R(:,:,2) = imfilter(double(Image), Gy, 'conv', 'replicate');
        R(:,:,3) = cosd(45)*R(:,:,1)+sind(45)*R(:,:,2);
        R(:,:,4) = cosd(135)*R(:,:,1)+sind(135)*R(:,:,2);
        R(:,:,5) = cosd(180)*R(:,:,1)+sind(180)*R(:,:,2);
        R(:,:,6) = cosd(225)*R(:,:,1)+sind(225)*R(:,:,2);
        R(:,:,7) = cosd(270)*R(:,:,1)+sind(270)*R(:,:,2);
        R(:,:,8) = cosd(315)*R(:,:,1)+sind(315)*R(:,:,2);
        FM = max(R,[],3);
        fm = mean2(FM);
    end
    function result = abslog2(x)
        if x > 0
        result = log2(x);
        elseif x < 0
            result = log2(-x);
        elseif x == 0
            result = 0;
        end
    end
end

    
