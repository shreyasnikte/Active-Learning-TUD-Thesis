import numpy as np
import pandas as pd
import random
import time
from multiprocessing import Pool

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import xgboost as xgb

# For data visualization
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from bokeh.io import output_notebook, show
from bokeh.models import Title
from bokeh.plotting import figure, output_file, show

import seaborn as sns

from datetime import datetime, timedelta, date
from tqdm import tqdm            #for .py version
# from tqdm import tqdm_notebook as tqdm     # for .ipynb version

pd.options.mode.chained_assignment = None  # default='warn'



# The dict 'params' consists of all the parameters used in the simulation software for ease of alteration
params = {
#         Set the regression model related parameters
          'train_start_dt':'2013-01',
          'train_stop_dt':'2013-12',
          'y_variable': 'trial_d',
          'X_variables':['trial_n', 'low', 'normal', 'high', 'WIND_DIRECTION', 
                         'WIND_SPEED', 'VISIBILITY', 'MSL_PRESSURE',
                         'AIR_TEMPERATURE', 'DEWPOINT', 'WETB_TEMP', 
                         'STN_PRES', 'WMO_HR_SUN_DUR', 'hour', 'day'],
    
#         Set XGBoost regression parameters (for consumption model)
          'n_estimators': 2000,
          'early_stopping_rounds': 50,  #stop if 50 consequent rounds without decrease of error
          'verbose': False,             # Change verbose to True if you want to see it train
          'nthread': 4,
    
#         Set simulator parameters to default values
          'season': 3,
          'day_of_week': 3,
          'special_event': 0,
          'tariff_policy':[],
    
#         Set Occupant behaviour dynamics
          'active_users': 0.1,#.5,     # Set the % of users who are willing to engage in the experiments
          'avail_users': 0.1,#.5,       # Set the % of users who will be available to participate in specific experiment
          'user_latency': 0,         # Set the values which correspond to real life participation delay for users 
          'frac_users_exp':1,      # Fraction of users selected for a particular trial
          
#         Set parameters for active learning
          'total_experiments':100,#100, #Total number of experiments allowed per trial
          'init_samples': 10,#50,      # Set the initial random samples to be chosen
          'test_size':.3,           # Set test data size for splitting data in train-test
          'X_var_activeL':['expected', 'dow', 'season'],
          'y_var_activeL':'response'
         }







class ConsumptionModel(object):
    def __init__(self, df, params):
        self.df = df
        self.params = params
#         some variables

    def prep_data(self):
        self.df = self.df.dropna().copy()
        one_hot= pd.get_dummies(self.df['tariff'])
        one_hot_renamed = one_hot.rename(index=str, columns={0.0399:'low', 0.1176:'normal', 0.672:'high'}) 
        self.df = self.df.join(one_hot_renamed).drop('tariff', axis=1)
        
        self.df["hour"] = self.df.index.hour
        self.df["day"] = self.df.index.day
        self.df["month"] = self.df.index.month


    
    def train(self):
#         Complete the xgboost model on 2013 data
        self.X_train = self.df.loc[self.params["train_start_dt"]:self.params["train_stop_dt"],self.params["X_variables"]]
        self.y_train = self.df.loc[self.params["train_start_dt"]:self.params["train_stop_dt"],self.params["y_variable"]]
        self.X_test = self.df.loc[self.params["train_stop_dt"]:,self.params["X_variables"]]
        self.y_test = self.df.loc[self.params["train_stop_dt"]:,self.params["y_variable"]]

        self.xg_reg = xgb.XGBRegressor(n_estimators=self.params['n_estimators'], nthread = self.params["nthread"])
        self.xg_reg.fit(self.X_train, self.y_train,
                        eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                        early_stopping_rounds = self.params["early_stopping_rounds"],
                        verbose = self.params["verbose"])

#         Get feature importance chart
        return xgb.plot_importance(self.xg_reg, height=0.9) # Plot feature importance
      

    def test(self, X_test, tariff):
#         test the data points. Get the predictions
#         self.preds = self.xg_reg.predict(X_test)
        pass
        

    def entropy(self):
#         get entropy of each data point nad return the entropy dataframe
        pass








class Simulator:
    
    def __init__(self, df, df_weather, params):
        self.params = params
        self.df = df
        self.df_weather = df_weather
        active_users = int(len(df.columns)*self.params["active_users"])   # get no. of active users from input percentage
        self.active_users = random.sample(list(df.columns), active_users)
        self.noisy_tariff = {}
        self.spring = [3, 4, 5]
        self.summer = [6, 7, 8]
        self.autumn = [9, 10, 11]
        self.winter = [1, 2, 12]


    def select_day(self):
#         Get user ids of participating users
        self.fuzzy_participation()
        
#         Select the season
        if selectsample.params["season"] == -1:
            month = random.randrange(1,12)
        elif selectsample.params["season"] == 0:
            month = random.choice(self.spring)
        elif selectsample.params["season"] == 1:
            month = random.choice(self.summer)
        elif selectsample.params["season"] == 2:
            month = random.choice(self.autumn)
        elif selectsample.params["season"] == 3:
            month = random.choice(self.winter)
            
#         Select the day of week
        if selectsample.params["day_of_week"] == -1:
#             Select random day
            dow = random.randrange(0,7)
        else:
            dow = selectsample.params["day_of_week"] 
            
#         Select the random day from the entries which satisfy above conditions
        shortlist = self.df.loc[(self.df.index.month == month) & (self.df.index.dayofweek == dow), :].index
        day = random.choice(shortlist.day.values)
        year = random.choice(shortlist.year.values)
        timestamp = str(year)+"-"+str(month)+"-"+str(day)
        self.sample = self.df.loc[timestamp,self.avail_users]
        
        
        
    def random_day(self):
#         Get user ids of participating users
        self.fuzzy_participation()
    
#         Sample a random day timestamp
        shortlist = self.df.sample(axis = 0).index
        day = random.choice(shortlist.day.values)
        month = random.choice(shortlist.month.values)
        year = random.choice(shortlist.year.values)
        timestamp = str(year)+"-"+str(month)+"-"+str(day)
        self.sample = self.df.loc[timestamp,self.avail_users]
        
        
    
    def fuzzy_participation(self):
        avail_users = int(len(self.active_users)*self.params["avail_users"])
        self.avail_users = random.sample(self.active_users, avail_users)
    
    
    def auto_noise_addition(self, levels, constraints):
#         select the random users and their behaviour with random latency
        self.noisy_tariff["h1_start"] = [random.choice(range(constraints["h1_start"]-2, 
                                                             constraints["h1_start"]+int(trials_.duration/2))) for _ in range(len(self.avail_users))]
        self.noisy_tariff["h1_end"] = [random.choice(range(constraints["h1_end"]-int(trials_.duration/2), 
                                                           constraints["h1_end"]+2)) for _ in range(len(self.avail_users))]
    

    def tariff_policy(self, levels, constraints):
#         use variables from auto_noise_addition and input variables of this function to create a tariff policy 
#         for each participating user **Needs more attention
        self.auto_noise_addition(levels,constraints)
    
        self.d = np.ones((48, len(self.avail_users)))
        self.df_tariff = pd.DataFrame(data=self.d, columns = self.avail_users)
        for i in range(len(self.avail_users)):
            self.df_tariff.loc[self.noisy_tariff["h1_start"][i]:self.noisy_tariff["h1_end"][i], self.avail_users[i]] = 2

        self.df_tariff.index = self.sample.index
        
        
    
            
    def test(self):
        self.preds = self.sample['trial_n']
        self.preds.loc[self.sample['high']==1] = self.preds.loc[self.sample['high']==1]*0.9 #(1 - 9/(100*self.params['active_users']*self.params['avail_users']))
        
        
        
    def run(self):
#         FOR EACH USER, call test function of consumption model, get modified behaviour, return original data point and modified data point
        self.sample = self.sample.interpolate(method = 'linear', axis = 0).ffill().bfill()
        self.sample = self.sample.join(self.df_weather.loc[self.sample.index,:])
        df_response = pd.DataFrame()
        self.sample["hour"] = self.sample.index.hour
        self.sample["day"] = self.sample.index.day
        self.sample["month"] = self.sample.index.month
        
        list_ = [i for i in range(len(self.avail_users))]

        for i in list_:
            one_hot= pd.get_dummies(self.df_tariff[self.avail_users[i]])
            one_hot_renamed = one_hot.rename(index=str, columns={1.0:'normal', 2.0:'high', 3.0:'low'}) 
            self.sample = pd.concat([self.sample, one_hot_renamed], axis =1)
            self.sample["low"] = 0

            self.sample["trial_n"] = self.sample[self.avail_users[i]]
            
#             consumption_model.test(self.sample[self.params['X_variables']], one_hot_renamed)
            self.test()
#             df_response[self.avail_users[i]] = consumption_model.preds
            df_response[self.avail_users[i]] = self.preds
            self.sample = self.sample.drop(['low', 'normal', 'high', 'trial_n'], axis= 1)
            
        df_response['response']= df_response.mean(axis = 1)
        return df_response['response']
            
            


class activeLearner(object):
    
    def __init__(self, df_n, df_weath, params):
        self.df_n = df_n
        self.df_weath = df_weath
        self.params = params
        

    
    
    def get_random_samples(self):
        print("Generating initial random samples...")
        temp_df = pd.DataFrame(columns = ['expected', 'response', 'dow', 'season'])
        
        
#         select first random day of 48 data points
        sim.random_day()
        
#         Add contextual data in future for the particular day to self.df
        
    
#         Generate new tariff signals for one day
        level, constraints = self.get_random_tariff()
        
#         Get schocastic behaviour of users
        sim.tariff_policy(level, constraints)
    
    
        response = sim.run()
        response_max = response.max()     # Peak consumption as a response
                
        expected = sim.sample[sim.avail_users].mean(axis = 1).values
        expected_max = expected.max()     # peak expected consumption
        
        dow, season = self.get_features()
        
        self.df_al.loc[0] = [expected_max, response_max, dow, season]

            
#        Create n number of datapoints from simulator (n=self.params["init_samples"])
#        Create a list of 1 to n to include a progress bar
        
        list_ = [i for i in range(self.params["init_samples"])]

        for i in tqdm(list_):
            
            sim.random_day()
            # Add contextual data in future for the particular day to temp_df
            level, constraints = self.get_random_tariff()
            sim.tariff_policy(level, constraints)
            
            response = sim.run()
            response_max = response.max()
            
            expected = sim.sample[sim.avail_users].mean(axis = 1).values
            expected_max = expected.max()
            
            dow, season = self.get_features()
            
            temp_df.loc[0] = [expected_max, response_max, dow, season]
            
            self.df_al = pd.concat([self.df_al, temp_df], axis=0, sort=True)

        self.df_rand = self.df_al.copy()  
            
            
            
    def split_data(self):
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(self.df_al[self.params['X_var_activeL']], 
                                                            self.df_al[self.params['y_var_activeL']], 
                                                            test_size= self.params["test_size"])
        
        
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(self.df_rand[self.params['X_var_activeL']], 
                                                            self.df_rand[self.params['y_var_activeL']], 
                                                            test_size= self.params["test_size"])
        
        return X_train_a, X_test_a, y_train_a, y_test_a, X_train_rand, X_test_rand, y_train_rand, y_test_rand 
        
        
        
    def get_features(self):
        try:
#             get day of week encoding
            dow = sim.sample.index[0].dayofweek
#             get month of year encoding
            month = sim.sample.index[0].month
#             we are more interested in season based behaviour than monthly behaviour
            
            if month in [3,4,5]:
                season =0
            elif month in [6,7,8]:
                season = 1
            elif month in [9,10,11]:
                season = 2
            elif month in [1,2,12]:
                season = 3
            
        except Exception as e: print(e)    

        return dow, season
        
    def run_random_forest_ActiveL(self, X_train, X_test, y_train):
        self.regres = RandomForestRegressor(max_depth=2, 
                                                random_state=0, 
                                                n_estimators=100)
        self.regres.fit(X_train, y_train)
        test_y_predicted = self.regres.predict(X_test)
        return (test_y_predicted)
    
    
    def run_random_forest_rand(self, X_train, X_test, y_train):
        self.regres = RandomForestRegressor(max_depth=2, 
                                                random_state=0, 
                                                n_estimators=100)
        self.regres.fit(X_train, y_train)
        test_y_predicted = self.regres.predict(X_test)
        return (test_y_predicted)
    

        
        
    def get_random_tariff(self):
        self.year = random.randrange(2012,2013)
        self.month = random.randrange(1,12)
        self.day = random.randrange(1,28)
        self.hour = random.randrange(17,18)
        self.minute = random.choice([0,30])
        self.duration = random.randrange(6, 8)
        index = datetime(self.year, self.month, self.day, self.hour, self.minute, 0)
        h1_start = int(index.hour * 2) + int(index.minute / 30)
        h1_end = h1_start + self.duration
        constraints = {"h1_start": h1_start, "h1_end": h1_end}
        level = 0       #dummy
        return level, constraints
        
        
        
        
    def run(self):
        self.df_al = pd.DataFrame(columns = ['expected', 'response', 'dow', 'season'])
        self.df_rand = pd.DataFrame(columns = ['expected', 'response', 'dow', 'season'])
        
        self.get_random_samples()
        
        print("Starting the trials...")
        mse_ActiveL = []
        mse_Rand = []
        
        
#         Create for loop to train the model for m number of times (where, m = self.params["total_experiments"] - self.params["init_samples"])
        
    
        list_ = [i for i in range(self.params["total_experiments"] - self.params["init_samples"])]
        for exp in tqdm(list_):
        
            # split data
            X_train_AL, X_test_AL, y_train_AL, y_test_AL, X_train_rand, X_test_rand, y_train_rand, y_test_rand= self.split_data()
            
            # train the models
            # model for Active learning
            preds_ActiveL = self.run_random_forest_ActiveL(X_train_AL, X_test_AL, y_train_AL)
            # model for Random sampling
            preds_rand = self.run_random_forest_rand(X_train_rand, X_test_rand, y_train_rand)

            
            
            # Query new data point
            # New data point by selective sampling
            sample_ActiveL, mse_activeL = selectsample.from_oracle(preds_ActiveL, X_test_AL, y_test_AL)
            # New data point by random sampling
            sample_rand, mse_rand = selectsample.from_oracle(preds_rand, X_test_rand, y_test_rand)

            
            # Add new data point to the existing data
            # Dataset with selectively sampled data points
            self.df_al = pd.concat([self.df_al, sample_ActiveL], axis=0, sort=True)
            # Dataset of randomly sampled datapoints
            self.df_rand = pd.concat([self.df_rand, sample_rand], axis=0, sort=True)

            
            mse_ActiveL.append(mse_activeL)
            mse_Rand.append(mse_rand)
            
            # find entropy (optional)
        return mse_ActiveL, mse_Rand
            # for next experiment, get tariff policy, season and weekday 
     
    

    
class SelectSample(object):
    def __init__(self, params):
        self.params = params
        
    def from_oracle(self, preds, X_test, y_test):
        #Select the point with maximum error
        df_y_test = y_test.reset_index()
        d = {'preds': preds}
        df_preds = pd.DataFrame(data = d)
        df_X_test = X_test.reset_index()
        
        error_ = (df_y_test['response']-df_preds['preds'])**2
        
        mse = ((df_y_test['response']-df_preds['preds'])**2).mean(axis=0)
        
        self.params["day_of_week"] = df_X_test.loc[error_.idxmax(),'dow']
        self.params["season"] = df_X_test.loc[error_.idxmax(),'season']
        
        # Generate new data point for above dow and season
        
        sim.select_day()
        level, constraints = trials_.get_random_tariff()
        sim.tariff_policy(level, constraints)
            
        response = sim.run()
        response_max = response.max()
          
        expected = sim.sample[sim.avail_users].mean(axis = 1).values
        expected_max = expected.max()
          
        dow, season = trials_.get_features()
    
        df = pd.DataFrame(columns = ['expected', 'response', 'dow', 'season'])
        df.loc[0] = [expected_max, response_max, dow, season]
        return df, mse
        
    def random(self):
        #Randomly select next data point
        sim.random_day()
        level, constraints = trials_.get_random_tariff()
        sim.tariff_policy(level, constraints)
            
        response = sim.run()
        response_max = response.max()
          
        expected = sim.sample[sim.avail_users].mean(axis = 1).values
        expected_max = expected.max()
          
        dow, season = trials_.get_features()
    
        df = pd.DataFrame(columns = ['expected', 'response', 'dow', 'season'])
        df.loc[0] = [expected_max, response_max, dow, season]
        return df, mse
    
    

        
        
        
        
        

def import_data():
    try:
        print("Reading aggregate consumption data...")
        df=pd.read_csv('~/Documents/work/Active-Learning-TUD-Thesis/mod_datasets/aggregate_consumption.csv', sep=',', header=0, index_col=0, parse_dates=['GMT'], low_memory=False)
        df = df.drop_duplicates()
        print("Done")
        print("Reading weather data...")
        df_midas=pd.read_csv('~/Documents/work/Active-Learning-TUD-Thesis/mod_datasets/midas_weather.csv', sep=',', header=0, index_col=0, parse_dates=['GMT'], low_memory=False)
        df_midas_rs = df_midas.resample('30T').mean()
        df_interpolated = df_midas_rs.interpolate(method='linear')
        df_weather = df_interpolated.loc['2013-01':'2013-12',:]
        df_final = pd.concat([df,df_weather], axis=1)
        print("Done")
        print("Reading LCL consumption data...")
        df_n=pd.read_csv('~/Documents/work/Active-Learning-TUD-Thesis/UKDA-7857-csv/csv/data_collection/data_tables/consumption_n.csv', sep=',', header=0, index_col=0, parse_dates=['GMT'], low_memory=False)
        df_n = df_n.drop_duplicates()
        df_n = df_n.fillna(df_n.mean())
        df_weath = df_interpolated.copy()
        print("Done")
        
    except Exception as e: print(e)
        
    return df_final, df_n, df_weath




  
def _init():
    df_final, df_n, df_weath = import_data()
    
    try:
        cons_model = ConsumptionModel(df_final, params)
        sim = Simulator(df_n.loc['2012-05':, :], df_weath.loc['2012-05':, :], params)
#         generate =  randomGenerate(params)
        trials_ = activeLearner(df_n.loc['2012-05':, :], df_weath.loc['2012-05':, :], params)
        selectsample = SelectSample(params)
        
    except Exception as e: print(e)    
    
    return cons_model, sim, trials_, selectsample





def plot_bokeh(mse_ActiveL, mse_Rand, params):
    output_notebook()
    output_file("./temp/line.html") #Uncomment it to save the plot in html file
    list_ = [i for i in range(params["total_experiments"])]

    p=figure(plot_width=800, plot_height=400)
    p.line(list_, mse_ActiveL, line_width=1, color='blue')
    p.line(list_, mse_Rand, line_width=1, color='red')
    show(p)
    
    
    
    
    
if __name__ == '__main__':
#     # import data and declare classes
    cons_model, sim, trials_, selectsample= _init()

#     # start the simulator and active learning by membership query synthesis

    mse_ActiveL, mse_Rand = trials_.run()
    list_ = [i for i in range(len(mse_ActiveL))]
    d1 = {'0':mse_ActiveL}
    mse_AL_total = pd.DataFrame(data=d1)
    d2 = {'0':mse_Rand}
    mse_Rand_total = pd.DataFrame(data=d2)
    
    for i in range(3):
        print("Iteration ", i+2)
        mse_ActiveL, mse_Rand = trials_.run()
        mse_AL_total.loc[:,str(i+1)] = mse_ActiveL
        mse_Rand_total.loc[:, str(i+1)] = mse_Rand
        
    plot_bokeh(mse_AL_total.mean(axis=1), mse_Rand_total.mean(axis=1), params)
