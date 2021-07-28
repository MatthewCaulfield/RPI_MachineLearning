
% Matthew Caulfield 
% ECSE 4962 Homework 3
% Implementation of the PEGASOS algorithm
% 11/9/2020
% mysgdsvm(filename, k, numruns)
% inputs: filename the name of the file that the mnist-13 is contained in.
% k is the size of each minibatch.
% numruns is the number of runs. 
% Outputs: The average runtime. The standard deviation of the runtimes. 
% The primal objective function value for each iteration is saved in a 
% tmp file.

function mysgdsvm(filename, k, numruns)
    data = load(filename);
    [n,m] = size(data);
    %lam sets the lambda for the regularization parameter of the SVD 
    lam = 30;
    %T is the total number of iterations 
    T = 4000;
    
    %Array where the runtimes are stored. 
    runtimes = zeros(1,numruns);
    %Array where the primal values of each iteration is stored
    primalList = zeros(numruns,T);
    
    for j = 1:numruns
        %start the clock
        start = clock;
        %initialize the weights to 0
        W = zeros(1,m-1);
        
        for t = 1:T
            %creates the minibatch for each iteration
            if k == 1
                randNum = randi(n);
                At = data(randNum,:);
            else
                [row1, col1] = find(data(:,1)==1);
                [row3, col3] = find(data(:,1)==3);
                index1=row1(randperm(length(row1)));
                index3=row3(randperm(length(row3)));
                percent = k/2000;

                index1=index1(1:round(percent*length(index1)));
                index3=index3(1:round(percent*length(index3)));
                At = [data(index1,:); data(index3,:)];
            end
            Atcopy = At;
            Atcopy(Atcopy==3)=-1;
            Atplus = 0;
            
            %Checking if each sample in the minibatch should be in AT+ and
            % if so sums each samples y*x
            for i = 1:k
                currSample =  Atcopy(i,:);
                y = currSample(1);
                x = currSample(2:end);
                check = y*dot(W,x);
                if check<1
                    Atplus = Atplus + x.*y;
                end
            end
            
            nt = 1/(lam*t);
            
            %updating the weights to their half step
            wthalf = (1-nt*lam)*W+(nt/k)*Atplus;
            
            %updating weights to their next value
            W = min(1, (1/(sqrt(lam)))/(norm(wthalf)))*wthalf;
            
            %calculating the loss for the current weights
            loss = 0;
            for i = 1:k
                    currSample =  Atcopy(i,:);
                    y = currSample(1);
                    x = currSample(2:end);
                    loss = loss + max(0,1-y*dot(W,x));
            end

            %calculates the primal for this iteration 
            primal = (lam/2)*(norm(W)^2) + (1/k)*loss;
            primalList(j,t) = primal;
            
        end
        
        stop = clock;
        runtimes(1,j) = etime(stop, start);
        
    end
% Plot the primitive function value
%     figure
%     for j = 1:numruns
%         plot(primalList(j,:), 'DisplayName',"run " + num2str(j))
%         hold on
%     end
%     xlabel('iterations')
%     ylabel('primal')
%     title("k = " + num2str(k))
%     legend show

    avgTime = mean(runtimes);
    stdTime = std(runtimes);
    writematrix(primalList, "./tmp.txt")
    fprintf("Avg runtime for " + num2str(numruns) + " runs with minibatch size of " + num2str(k) + ": " + num2str(avgTime) +"sec.\n")
    fprintf("Std runtime for " + num2str(numruns) + " runs with minibatch size of " + num2str(k) + ": " + num2str(stdTime)+"sec.\n")
    fprintf("Plot data exported to ./tmp.txt\n")
end

