import numpy as np
import pandas as pd
import random
import time
from multiprocessing import Pool


from datetime import datetime, timedelta, date
from tqdm import tqdm            #for .py version
# from tqdm import tqdm_notebook as tqdm     # for .ipynb version





# The dict 'params' consists of all the parameters used in the simulation software for ease of alteration
params = {    
    
#         Set simulator parameters to default values
          'season': 3,
          'day_of_week': 3,
          'special_event': 0,
          'tariff_policy':[],
    
#         Set Occupant behaviour dynamics
          'active_users': 0.5,#.5,     # Set the % of users who are willing to engage in the experiments
          'avail_users': 0.8,#.5,       # Set the % of users who will be available to participate in specific experiment
          'user_latency': 0,         # Set the values which correspond to real life participation delay for users 
          'frac_users_exp':1,      # Fraction of users selected for a particular trial
          
#         Set parameters for active learning
          'total_iterations':20000,
          
          'X_var_activeL':['dow', 
                           'season', 
                           'hod', 
                           'AIR_TEMPERATURE', 
                           'DEWPOINT', 
                           'MSL_PRESSURE', 
                           'STN_PRES',
                           'VISIBILITY', 
                           'WETB_TEMP',
                           'WIND_DIRECTION',
                           'WIND_SPEED',
                           'WMO_HR_SUN_DUR',
                           'hod', 
                           'month'],
    
          'y_var_activeL':'expected'
         }















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
        
#         day = random.choice(shortlist.day.values)
#         year = random.choice(shortlist.year.values)
        random_index = random.choice(shortlist)
        timestamp = str(random_index.year)+"-"+str(random_index.month)+"-"+str(random_index.day)
#         print(timestamp, " Select day")
        self.sample = self.df.loc[timestamp,self.avail_users]

        
        
        
        
    def random_day(self):
#         Get user ids of participating users
        self.fuzzy_participation()
    
#         Sample a random day timestamp
        shortlist = self.df.sample(axis = 0).index
#         day = random.choice(shortlist.day.values)
#         month = random.choice(shortlist.month.values)
#         year = random.choice(shortlist.year.values)
        random_index = random.choice(shortlist)
        self.timestamp = str(random_index.year)+"-"+str(random_index.month)+"-"+str(random_index.day)
#         print(timestamp, " Random day")
        self.sample = self.df.loc[self.timestamp,self.avail_users]
        
        
    
    def fuzzy_participation(self):
        avail_users = int(len(self.active_users)*self.params["avail_users"])
        self.avail_users = random.sample(self.active_users, avail_users)
    
    
    def auto_noise_addition(self, levels, constraints):
#         select the random users and their behaviour with random latency
        self.noisy_tariff["h1_start"] = [random.choice(range(constraints["h1_start"]-2, 
                                                             constraints["h1_start"]+int(self.duration/2))) for _ in range(len(self.avail_users))]
        self.noisy_tariff["h1_end"] = [random.choice(range(constraints["h1_end"]-int(self.duration/2), 
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
            
            
            
    def test(self):
        self.preds = self.sample['trial_n']
        self.preds.loc[self.sample['high']==1] = self.preds.loc[self.sample['high']==1]*0.9 #(1 - 9/(100*self.params['active_users']*self.params['avail_users']))
        
        















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
        df_weath = df_interpolated.copy()
        print("Done")
        
    except Exception as e: print(e)
        
    return df_final, df_n, df_weath







def _init():
    df_final, df_n, df_weath = import_data()
    
    try:
        sim = Simulator(df_n.loc['2012-05':, :], df_weath.loc['2012-05':, :], params)
        
    except Exception as e: print(e)    
    
    return sim, df_n.loc['2012-05':, :], df_weath.loc['2012-05':, :]


def get_features():
    try:
#             get day of week encoding
        dayofweek = sim.sample.index.dayofweek
#             get month of year encoding
        month = sim.sample.index.month
#             get hour of day value from timestamp
        hourofday = sim.sample.index.hour
#             we are more interested in season based behaviour than monthly behaviour
            
        season = [0 if x in [3,4,5] else x for x in month]
        season = [1 if x in [6,7,8] else x for x in season]
        season = [2 if x in [9,10,11] else x for x in season]
        season = [3 if x in [1,2,12] else x for x in season]
        
        
        
                    
    except Exception as e: print(e)    

    return dayofweek, month, hourofday, season





def get_random_tariff():
    year = random.randrange(2012,2013)
    month = random.randrange(1,12)
    day = random.randrange(1,28)
    hour = random.randrange(17,18)
    minute = random.choice([0,30])
    duration = random.randrange(6, 8)
    index = datetime(year, month, day, hour, minute, 0)
    h1_start = int(index.hour * 2) + int(index.minute / 30) 
    h1_end = h1_start + duration
    constraints = {"h1_start": h1_start, "h1_end": h1_end}
    level = 0       #dummy
    return level, constraints
    




if __name__ == '__main__':
#     # import data and declare classes
    sim, df_n, df_weather = _init()


#     # start the data generator
    temp_df = pd.DataFrame(columns = ['expected', 
                                          'response', 
                                          'dow', 
                                          'season'])
    
    
#   select first random day of 48 data points
    sim.random_day()
    
#   Add contextual data in future for the particular day to self.df
    

#   Generate new tariff signals for one day
    level, constraints = sim.get_random_tariff()
    
#   Get schocastic behaviour of users
    sim.tariff_policy(level, constraints)


    response = sim.run()
            
    expected = sim.sample[sim.avail_users].mean(axis = 1).values
    
    dayofweek, month, hourofday, season = get_features()
        
    data = {'expected':expected, 
            'response':response.values, 
            'dow':dayofweek, 
            'season':season,
            'hod': hourofday,
            'month': month}

    df_ = pd.DataFrame(data, index=response.index)
    df = pd.concat([df_,df_weather.loc[sim.timestamp,:]], axis=1)

        
#        Create n number of datapoints from simulator (n=self.params["init_samples"])
#        Create a list of 1 to n to include a progress bar
    
    list_ = [i for i in range(params["total_iterations"])]

    for i in tqdm(list_):
        
        sim.random_day()
        
#             Decide the tariff signal and stochastic behaviour of users around that tariff signal
        level, constraints = get_random_tariff()
        sim.tariff_policy(level, constraints)
        
        response = sim.run() 
        
        expected = sim.sample[sim.avail_users].mean(axis = 1).values
        
        dayofweek, month, hourofday, season = get_features()
        
        data = {'expected':expected, 
                'response':response.values, 
                'dow':dayofweek, 
                'season':season,
                'hod': hourofday,
                'month': month}
        
        
    
        df_ = pd.DataFrame(data, index=response.index)
        temp_df = pd.concat([df_,df_weather.loc[sim.timestamp,:]], axis=1)
        df = pd.concat([df, temp_df], axis=0, sort=True)
        


