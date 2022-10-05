for ii = 1:12
    
Vertices = [];
Angles = [];

parfor j = 1:100000

N = 5;   % Number of vertices
generation = true;

while generation    
    %% Generate the polar angles and radii of the vertices
        % We first randomly generate the polar angles. The first angle is
        % fixed at zero degree. The rest angles are generated one by one.
        % During generation we strictly limit the origin to be inside the
        % polygon so that we have a direct control of its size.
    
    spacing = pi/10;   % The minimum spacing between neighboring polar angles. This ensures the edges to be visible

    Angle = zeros(1,N);
    Angle(2) = (Angle(1)+spacing) + (pi-Angle(1)-spacing)*rand;   % The second polar angle. We limit the angle range such that the origin is inside the polygon.
    for i = 3:N
        if i ~= N
            Angle(i) = (Angle(i-1)+spacing) + min([(2*pi-(N+1-i)*spacing)-Angle(i-1)-spacing, pi-spacing])*rand;   % The third to the second last polar angle. 
            % The min function ensures the polar angle spacing and puts the origin inside the polygon.

        elseif i == N   % If generating the last polar angle

            if Angle(i-1) + spacing < pi
                MAX = min([(2-spacing)*pi-pi, Angle(i-1)]);   % The min function puts the origin inside the polygon.
                Angle(i) = pi + min([MAX, pi])*rand;
            else
                Angle(i) = (Angle(i-1)+spacing) + min([(2*pi-(N+1-i)*spacing)-Angle(i-1)-spacing, pi-spacing])*rand;   % The min function ensures the polar angle spacing and puts the origin inside the polygon.
            end
        end

    end


    rho_max = 2;
    rho_min = 0.5;

    Radii = rho_min + (rho_max-rho_min)*rand(1,N);   % Generate associated polar radii

    %% Gauge fixing
    
    [x, y] = pol2cart(Angle, Radii);   % Convert to Cartesian 

    Area = 1/2*(sum(x.*circshift(y,-1,2))-sum(circshift(x,-1,2).*y));
    C_x = 1/(6*Area)*sum((x+circshift(x,-1,2)).*(x.*circshift(y,-1,2)-circshift(x,-1,2).*y));
    C_y = 1/(6*Area)*sum((y+circshift(y,-1,2)).*(x.*circshift(y,-1,2)-circshift(x,-1,2).*y));   % Find the centroid of the polygon
    
    Angle_2 = [Angle(2:N),2*pi];
    InnerAngle = Angle_2-Angle;   % The angle between neighboring polar angles
    
    CircRadii = circshift(Radii,-1,2);
    Edge = sqrt(Radii.^2 + CircRadii.^2 - 2 * Radii.*CircRadii.*cos(InnerAngle));   % Find the edges of the polygon

    EdgeAngle_1 = acos((Radii.^2 + Edge.^2 - CircRadii.^2)./(2 * Radii.*Edge));
    EdgeAngle_2 = acos((CircRadii.^2 + Edge.^2 - Radii.^2)./(2 * CircRadii.*Edge));

    EdgeAngle = EdgeAngle_1 + circshift(EdgeAngle_2,1,2);   % Find the innerangles of the polygon

    temp = [EdgeAngle;Edge;x;y];
    angle_min = min(temp(1,:));
    position = find(temp(1,:)==angle_min);   % Position of the minimum innerangle

    temp_1 = circshift(temp,-1*(position-1),2);   % Shift the minimum innerangle to be the first
    compare = temp_1(2,1)<=temp_1(2,end);

    if compare~=1  
        temp_2 = [[temp_1(1,1),flip(temp_1(1,2:end))];flip(temp_1(2,:));[temp_1(3,1),flip(temp_1(3,2:end))];[temp_1(4,1),flip(temp_1(4,2:end))]];   % Switching the direction of traversing the polygon
    else
        temp_2 = temp_1;
    end

    x_centroid = temp_2(3,:) - C_x;   % Move the centorid to the origin
    y_centroid = temp_2(4,:) - C_y;
    EdgeAngle = temp_2(1,:);
    
    rot_angle = atan(y_centroid(1)/x_centroid(1));   % Find the angle of rotation
    if x_centroid(1) < 0 && rot_angle < 0
        rot_angle = rot_angle + pi;
    elseif x_centroid(1) < 0 && rot_angle > 0
        rot_angle = rot_angle + pi;
    end
    rot_matrix = [[cos(rot_angle), sin(rot_angle)]; [-sin(rot_angle), cos(rot_angle)]];
% 
    vertices_centroid = rot_matrix*[x_centroid(2:end); y_centroid(2:end)];
    vertices_centroid = [[sqrt(x_centroid(1)^2+y_centroid(1)^2);0],vertices_centroid];   % Rotate the smallest innerangel to be along the positive x-axis

    vertices_centroid = round(vertices_centroid,1);   % put the vertices into a square grid with 0.1 spacing. The grid is from -2 to 2
    
    if max(abs(vertices_centroid(:)))>2   % If one vertex is outside the grid, start over
        continue
    end

    gd = [2 N vertices_centroid(1,:) vertices_centroid(2,:)]';
    a = csgchk(gd);   % Check for self-intersection
    
    if a == 0   % If good, store vertices and innerangles

        Vertices = [Vertices; [vertices_centroid(1,:), vertices_centroid(2,:)]];
        Angles = [Angles; EdgeAngle];
 
        generation = false;
        
    end
    
end
end

dd = strcat('2D_Matrix_Eig_5_polygon_Mathematica_no_reflection_',num2str(ii),'.h5');   % Save the data
h5create(dd, '/Vertices', [100000 10])
h5write(dd, '/Vertices', Vertices)
    
h5create(dd, '/Angles', [100000 5])
h5write(dd, '/Angles', Angles)
end