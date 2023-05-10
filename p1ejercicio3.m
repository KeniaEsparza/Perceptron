clear; 
%% Carga de entradas y salidas de entrenamiento
datos = csvread('concentlite.csv');
x = datos(:,[1,2]);
t = datos(:,[3]);
%% Declaración de la red neuronal multicapas 
net = feedforwardnet(10,'trainlm');

net = train(net,x',t');

%% Evaluación de la red
y = net(x');

%% Impresión de gráficas
figure();
hold on;
unique_TargetClasses = unique(t);
training_colors = {'r.', 'b.'};
separation_colors = {'g.', 'm.'};

for i = 1:length(unique_TargetClasses)
    points = x(t==unique_TargetClasses(i), 1:end);
    plot(points(:,1), points(:,2), training_colors{i}, 'markersize', 10);
end
figure();
hold on;
for i=1:length(x')
    if (y(i) > 0.6) %TODO: Not generic role for any number of output nodes
        plot(x(i,1)', x(i,2)', separation_colors{1}, 'markersize', 10);
    else
        plot(x(i,1)', x(i,2)', separation_colors{2}, 'markersize', 10);
    end
end
