from metaflow import FlowSpec,step,Parameter,IncludeFile
from data_generator import dfGenerator

def script_path(filename):
    import os
    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)

class ModelTest(FlowSpec):

    csv_data = IncludeFile("csv_data",help="The path of the houses data",default=script_path("USA_Housing.csv"))
    name = Parameter("name",help="country name",default="USA")
    
    @step
    def start(self):
        print("modelFlow is starting!")
        self.next(self.collect_data)

    @step
    def collect_data(self):
        import pandas as pd
        from io import StringIO

        self.df = pd.read_csv(StringIO(self.csv_data))
        print("DataSet imported with sucess")
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        from sklearn.model_selection import train_test_split
        import pandas as pd

        # define arrays X and y
        self.X = self.df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
        self.y = self.df['Price']

        # Define datas to Train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.4, random_state = 404)

        print("train and test datas are created")
        self.next(self.modelcreate)


    @step
    def modelcreate(self):
        from sklearn.linear_model import LinearRegression

        # Creating model and training 
        lm = LinearRegression().fit(self.X_train,self.y_train)


        print("Intercept = " + str(lm.intercept_))
        print("\nCoeficients:")
        for i,name in enumerate(self.X):
            print(name + " = " + str(lm.coef_[i]))

        self.predictions = lm.predict(self.X_test)
        
        
        self.next(self.calc_mae,self.calc_mse,self.calc_rmse)


    @step
    def calc_mae(self):
        from sklearn import metrics

        self.MAE = metrics.mean_absolute_error(self.y_test, self.predictions)
        print("MAE measured")

        self.next(self.join)

    @step
    def calc_mse(self):
        from sklearn import metrics

        self.MSE = metrics.mean_squared_error(self.y_test, self.predictions)
        print("MSE measured")
        self.next(self.join)

    @step
    def calc_rmse(self):
        from sklearn import metrics
        import numpy as np

        self.RMSE = np.sqrt( metrics.mean_squared_error(self.y_test, self.predictions))
        print("RMSE measured")
        self.next(self.join)


    @step
    def join(self,inputs):
        #  This step is to collect all the results and save in one variable
        print("Results:")
        print("MAE = " + str(inputs.calc_mae.MAE))
        print("MSE = " + str(inputs.calc_mse.MSE))
        print("RMSE = " + str(inputs.calc_rmse.RMSE))
        self.measures = {"mae":inputs.calc_mae.MAE,"mse": inputs.calc_mse.MSE,"rmse" : inputs.calc_rmse.RMSE}
        self.compare = {"results" : inputs.calc_mae.predictions ,"answers" : inputs.calc_mae.y_test}
        self.next(self.end)

    @step
    def end(self):
        #end of this flow
        print("Done!")

if(__name__ == '__main__'):
    ModelTest()


