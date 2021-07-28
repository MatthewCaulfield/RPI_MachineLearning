Matthew Caulfield
RIN 661485877
caulfm@rpi.edu

To run this code you can run main.py after setting fileName to the 
file and path of the corresponding data set, setting numberSplits to 
the number of trials and training percents to the vector containg percentage
of the training data. If line 34 of logisiticRegression.py and line 42
of naiveBayesGaussian are uncommented then the error data for 
each trial will be saved in csv files in the folder LRoutput.csv
and output.csv respectivly. The output of each function is a table 
containing the error results for each trial in the for

%training data
		5    10   15   20   25   30
 trial number  
	1
	2
	3
	4

To run logisitic regression call logisticRegression(filename,
num splits, train percent)
To run just naives bayes gaussian call naiveBayesGaussian(filename,
num splits, train percent) 