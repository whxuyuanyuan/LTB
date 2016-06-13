threshold 1 = 0.15; % threshold for converting image to binary image
for step = 0: 100    
    imgUv = imread(strcat('pictures/', num2str(step, '%04d'), '_Uv.jpg'));
    imgUv = imgUv(:, :, 3); % blue channel
    imgUvBin = im2bw(imgUv, threshold1);
    %imshow(imgUvBin)

    centerProps = regionprops(imgUvBin, 'Centroid', 'Area', 'Eccentricity');

    % x, y-coordinates of the center of each circle
    center_x = [];
    center_y = [];

    % x, y-coordinates of the center of each rectangle
    rect_x = [];
    rect_y = [];

    % Loop over each region and filter the fake.
    for i = 1: length(centerProps)
        % the range of local region is [260, 1500]
        if centerProps(i).Area >= 260 && centerProps(i).Area <= 1500 
            % upper bound of the eccentricity for a circle is 0.7
            if centerProps(i).Eccentricity < 0.7
                center_x = cat(1, center_x, centerProps(i).Centroid(1));
                center_y = cat(1, center_y, centerProps(i).Centroid(2));
            else
                rect_x = cat(1, rect_x, centerProps(i).Centroid(1));
                rect_y = cat(1, rect_y, centerProps(i).Centroid(2));
            end
        end
    end

    % Array of angles
    orien = []; 

    % Loop over each circle
    for i = 1: length(center_x)
        minVal = 10000; % min value of the distance bewteen the circle and the rectangle
        index = 1; % temp value
        % loop over each rectangle (dirty method) and find the nearest distance
        for j = 1: length(rect_x)
            dist2 = (rect_x(j) - center_x(i))^2 + (rect_y(j) - center_y(i))^2;
            if dist2 < minVal
                minVal = dist2;
                index = j;
            end
        end
        orien = cat(1, orien, atan2(rect_y(index) - center_y(i), rect_x(index) - center_x(i)));
    end

    % Plot
    %{
    for i = 1: length(center_x)
        hold on
        scatter(center_x(i), center_y(i), 'filled')
    end
    %}
    
    % Save data
    f = fopen(strcat('roughposition/step_', num2str(step, '%04d')), 'w');
    for i = 1: length(center_x)
        fprintf(f, '%12f %12f %12f\n', center_x(i), center_y(i), orien(i));
    end
    fclose(f);
end
















