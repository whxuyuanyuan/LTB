for step = 0: 100
    imgUv = imread(strcat('pictures/', num2str(step, '%04d'), '_Uv.jpg'));
    %imgUv = imgUv(:, :, 3);
    imgUvBin = im2bw(imgUv, 0.018);
    %imshow(imgUvBin)
    centerProps = regionprops(imgUvBin, 'Centroid', 'Area', 'Eccentricity');

    center_x = [];
    center_y = [];

    center_hep_x = [];
    center_hep_y = [];

    center_pen_x = [];
    center_pen_y = [];

    rect_x = [];
    rect_y = [];

    for i = 1: length(centerProps)
        if centerProps(i).Area >= 230 && centerProps(i).Area <= 1500 && centerProps(i).Centroid(2) > 70
            if centerProps(i).Eccentricity < 0.7
                center_x = cat(1, center_x, centerProps(i).Centroid(1));
                center_y = cat(1, center_y, centerProps(i).Centroid(2));
            else
                rect_x = cat(1, rect_x, centerProps(i).Centroid(1));
                rect_y = cat(1, rect_y, centerProps(i).Centroid(2));
            end
        end
    end

    orien = [];

        for i = 1: length(center_x)
            minVal = 10000;
            index = 1;
            for j = 1: length(rect_x)
                dist2 = (rect_x(j) - center_x(i))^2 + (rect_y(j) - center_y(i))^2;
                if dist2 < minVal
                    minVal = dist2;
                    index = j;
                end
            end
            if minVal < 2500
                center_hep_x = cat(1, center_hep_x, center_x(i));
                center_hep_y = cat(1, center_hep_y, center_y(i));
                orien = cat(1, orien, atan2(rect_y(index) - center_y(i), rect_x(index) - center_x(i)));
            else
                center_pen_x = cat(1, center_pen_x, center_x(i));
                center_pen_y = cat(1, center_pen_y, center_y(i));
            end
        end

    if length(center_hep_x) ~= 331 || length(center_pen_x) ~= 331
        fprintf('%d\n', step);
    end
    
    for i = 1: length(center_pen_x)
	    hold on
	    scatter(center_pen_x(i), center_pen_y(i), 'filled', 'r');
    end
    
	for i = 1: length(center_hep_x)
	    hold on
	    scatter(center_hep_x(i), center_hep_y(i), 'filled', 'b');
    end
    
    
    f = fopen(strcat('rough_position/step_', num2str(step, '%04d')), 'w');
    for i = 1: length(center_pen_x)
        fprintf(f, '%12f %12f\n', center_pen_x(i), center_pen_y(i));
    end

    for i = 1: length(center_hep_x)
        fprintf(f, '%12f %12f %12f\n', center_hep_x(i), center_hep_y(i), orien(i));
    end
end





            
           
            
        

 
