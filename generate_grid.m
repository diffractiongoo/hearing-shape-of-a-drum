function grid_matrix = generate_grid(N,N_grid,vertices_centroid)   % Only optimized for 41x41 grid
vertices_centroid(1,:) = 10 * vertices_centroid(1,:) + 21;   % Convert the coordinates from [-2, 2] to [1, 41]
vertices_centroid(2,:) = -10 * vertices_centroid(2,:) + 21;
vertices_centroid = [vertices_centroid, vertices_centroid(:,1)];
index_X_1 = [];
index_X_2 = [];
for j = 1:N   % Draw boundaries on the grid
    X_1 = linspace(vertices_centroid(1,j),vertices_centroid(1,j+1),abs(vertices_centroid(1,j+1)-vertices_centroid(1,j))+1);   % Horizontal sweep
    k_1 = (vertices_centroid(2,j+1)-vertices_centroid(2,j))/(vertices_centroid(1,j+1)-vertices_centroid(1,j));
    b_1 = vertices_centroid(2,j)-k_1*vertices_centroid(1,j);
    Y_1 = round(k_1*X_1+b_1);   % y-coordinates of the boundaries
    index_X_1 = [index_X_1, (X_1(1:end-1)-1)*N_grid+Y_1(1:end-1)];   % index of the matrix elements that are on the boundary

    X_2 = linspace(vertices_centroid(2,j),vertices_centroid(2,j+1),abs(vertices_centroid(2,j+1)-vertices_centroid(2,j))+1);   % Vertical sweep
    k_2 = (vertices_centroid(1,j+1)-vertices_centroid(1,j))/(vertices_centroid(2,j+1)-vertices_centroid(2,j));
    b_2 = vertices_centroid(1,j)-k_2*vertices_centroid(2,j);
    Y_2 = round(k_2*X_2+b_2);   % x-coordinates of the boundaries
    index_X_2 = [index_X_2, (Y_2(1:end-1)-1)*N_grid+X_2(1:end-1)];   % index of the matrix elements that are on the boundary
end
grid_matrix = zeros(N_grid,N_grid);
grid_matrix(index_X_1) = 1;
grid_matrix(index_X_2) = 1;

end