clc
clear all

%% load the Iris Dataset and compute the input and Target 

[X,y]=iris_dataset;  % load the iris dataset

yv=[];
for i =1:size(y,2)
     yv(i)=find(y(:,i)==1);
end

X=X';
y=yv';

%% split the dataset into 80% training and 20% testing
testsize =0.8;  % test size

idx=randperm(length(X));  % create list of random permutation
n=floor(testsize*length(idx));  % compute the number of data record in test size

Xtrain=X(idx(1:n),:);   % extract the training input
ytrain=y(idx(1:n));     % extract the training target class

Xtest=X(idx(n+1:end),:);  % extract the testing input
ytest=y(idx(n+1:end));    % extract the testing target

%% Initialize Training parameters matrix to store pior and covariance information

[n,m]=size(Xtrain);   % compute the test size
classes=unique(y);    % extract the classes
nclass=length(classes);  % find the number of classes 
Msubdata = zeros(nclass,m);                    %Matrix to store the mean of subdata for each class
prob = zeros(nclass,1);                        %Matrix to store probabilities 
covinformation = zeros(m);  %Matrix to store covariance information

%% Training process

for c =1:nclass  % loop the classes to compute the pior and covariance information for each class
    
    subsetdata = Xtrain(ytrain==classes(c),:);  %subset data

    Input=subsetdata(:,1:end);  %extract the input for subset

    L=length(Input); % Length of subset data

    probs(c)=L/n;   % compute the pior and store the pior probability

    % calculate the mean vector
    meanvector = mean(Input);

    Msubdata(c,:) = meanvector; % store the mean for the subdata

    % calculate the covariance matrix 
    covmatrix = cov(Input);

    %update the covariance information
    covinformation = covinformation + ((L-1) / (n-nclass)).*covmatrix;

end

%% compute the beta parameter for each class
weight=zeros(c,m+1);  % %Matrix to store model coefficient
for c= 1:nclass  %loop through the number of classes

    % constant
    weight(c,1) = -0.5* (Msubdata(c,:)/ covinformation) * Msubdata(c,:)' + log(probs(c));

    % linear coefficient
    weight(c,2:end)= (Msubdata(c,:)/ covinformation);
end

predictions=[];  %list to store predictions
for i = 1: size(Xtest,1) %loop through the training data

    % extract the test sample
    testsample=Xtest(i,:);

    predlinear=sum(testsample.*weight(:,2:end),2) + weight(:,1);

    % find the maximum index
    [~,idx]=max(predlinear);

    % determine the classes, which is the maximum index
    predictions(i)=classes(idx);

end

cfm=confusionmat(ytest,predictions)

accuracy=100*sum(diag(cfm))/sum(cfm(:))