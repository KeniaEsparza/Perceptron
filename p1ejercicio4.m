clear;
% Paso 1: Cargar los datos del archivo irisbin.csv
datos = csvread('irisbin.csv');

% Paso 2: Separar los datos en características y etiquetas
caracteristicas = datos(:, 1:4)';
etiquetas = datos(:, 5:7)';

% 1 = Setosa, 2 = Versicolor, 3 = Virginica

for i=1:length(etiquetas)
    if(etiquetas(3,i)==1)
        etiquetas(1,i)=1;
    elseif(etiquetas(2,i)==1)
        etiquetas(1,i)=2;
    elseif(etiquetas(1,i)==1)
        etiquetas(1,i)=3;
    end
end

etiquetas = etiquetas(1,:);

%% Se separan los sets de entrenamiento con leave-k-out donde K es igual a 10
c = cvpartition(etiquetas,'k',10);
errorKout = zeros(c.NumTestSets,1);


%% Se define la red neuronal
red = patternnet(10); % Red neuronal de 10 neuronas en la capa oculta

%% Se realizan todos los sets de entrenamiento, y se calcula el error de cada uno de ellos
for i = 1:c.NumTestSets
    trainingIdx = c.training(i);
    testIdx = c.test(i);
    red = trainlm(red, caracteristicas(:,trainingIdx), etiquetas(trainingIdx)); 
    ytest = red(caracteristicas(:,testIdx));
    errorKout(i) = sum(ytest-etiquetas(testIdx));
end
%% Se promedia el error global, de acuerdo al número de sets de entrenamiento que se realizaron
crosvalidationErrorKout = sum(errorKout)/sum(c.TestSize);
standardDeviationKout = std(errorKout);

clear red;

%% Se separan los sets de entrenamiento con leave-k-out donde K es igual a 10
c = cvpartition(etiquetas,'Leaveout');
errorOneout = zeros(c.NumTestSets,1);

%% Se define la red neuronal
red = patternnet(10); % Red neuronal de 10 neuronas en la capa oculta

%% Se realizan todos los sets de entrenamiento, y se calcula el error de cada uno de ellos
for i = 1:c.NumTestSets
    trainingIdx = c.training(i);
    testIdx = c.test(i);
    red = trainlm(red, caracteristicas(:,trainingIdx), etiquetas(trainingIdx)); 
    ytest = red(caracteristicas(:,testIdx));
    errorOneout(i) = sum(ytest-etiquetas(testIdx));
end
%% Se promedia el error global, de acuerdo al número de sets de entrenamiento que se realizaron
crosvalidationErrorOneOut = sum(errorOneout)/sum(c.TestSize);
standardDeviationOneout = std(errorOneout);