for i = 1:12
    % Load data
    d = strcat('2D_Matrix_Eig_5_polygon_Mathematica_no_reflection_eig_',num2str(i),'.h5');
    dd = strcat('2D_Matrix_Eig_5_polygon_Mathematica_no_reflection_',num2str(i),'.h5');
    EigValue = h5read(d,'/EigValue');
    EigValue = transpose(EigValue);
    Vertices = h5read(dd,'/Vertices');
    Angles = h5read(dd,'/Angles');
    
    N = 5;   % Number of vertices
    N_grid = 41;   % Number of grid points
    Matrix = zeros(100000,41,41);   % Matrix of 100,000 unfilled polygons with only the boundaries
    Matrix_1 = zeros(100000,41,41);   % Matrix of 100,000 filled polygons
    
    for j = 1:100000
        vertices_centroid = [Vertices(j,1:N);Vertices(j,N+1:end)];
        A = generate_grid(N,N_grid,vertices_centroid);
        Matrix(j,:,:) = A;
        A = reshape(A,[41 41]);
        A = logical(A);
        xlimit = Vertices(j,1:N);
        ylimit = Vertices(j,N+1:end);
        [in, on] = inpolygon(0,0,xlimit,ylimit);
        if (in-on)==0   % If the origin is on the boundary
            xbox = [xlimit, xlimit(1)];
            ybox = [ylimit, ylimit(1)];

            x = [-2, 2];
            y = [0, 0];

            [xi,yi] = polyxpoly(x,y,xbox,ybox,'unique');
            xi = sort(xi,'ascend');
            midpoints = xi(1:end-1) + diff(xi)/2;
            midpoints = round(midpoints,1);
            [check_in, check_on] = inpolygon(midpoints,zeros(1,length(midpoints)),xlimit,ylimit);
            f = 0;
            for k = 1:length(check_in)
                if (check_in(k)-check_on(k))==1
                   x_start = 10 * midpoints(k) + 21;
                   AA = imfill(A, [21 x_start], 4);
                   AA = double(AA);
                   Matrix_1(j,:,:) = AA;
                   break
                end
                f = f+1;
            end
            if f==length(check_in)
                x = [0, 0];
                y = [-2, 2];

                [xi,yi] = polyxpoly(x,y,xbox,ybox,'unique');
                yi = sort(yi,'ascend');
                midpoints = yi(1:end-1) + diff(yi)/2;
                midpoints = round(midpoints,1);
                [check_in, check_on] = inpolygon(zeros(1,length(midpoints)),midpoints,xlimit,ylimit);
                f = 0;
                for k = 1:length(check_in)
                    if (check_in(k)-check_on(k))==1
                       y_start = -10 * midpoints(k) + 21;
                       AA = imfill(A, [y_start 21], 4);   % fill the polygon
                       AA = double(AA);
                       Matrix_1(j,:,:) = AA;
                       break
                    end
                    f = f+1;
                end
                if f==length(check_in)
                    fprintf(1,'found') 
                end
            end
        else   % If the origin is inside the polygon
            AA = imfill(A, [21 21], 4);   % fill the polygon
            AA = double(AA);
            Matrix_1(j,:,:) = AA;
        end
        
    end
    
    % Save the data separately
    ddd = strcat('2D_Matrix_Eig_5_polygon_Mathematica_no_reflection_filled_',num2str(i),'.h5');
    
    % save the unfilled polygons
    h5create(dd, '/EigValue', [100000 100])
    h5write(dd, '/EigValue', EigValue)
    
    h5create(dd, '/Matrix', [100000 41 41])
    h5write(dd, '/Matrix', Matrix)
    
    % save the filled polygons
    h5create(ddd, '/EigValue', [100000 100])
    h5write(ddd, '/EigValue', EigValue)
    
    h5create(ddd, '/Matrix', [100000 41 41])
    h5write(ddd, '/Matrix', Matrix_1)
    
    h5create(ddd, '/Vertices', [100000 10])
    h5write(ddd, '/Vertices', Vertices)

    h5create(ddd, '/Angles', [100000 5])
    h5write(ddd, '/Angles', Angles)

end