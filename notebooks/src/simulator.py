# This file contains the simulator for the dToU tariff responses. 
# More information will be avaiable in the wiki
# Author: Shreyas Nikte, TU Delft

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
          'active_users': .5,     # Set the % of users who are willing to engage in the experiments
          'avail_users': .8,       # Set the % of users who will be available to participate in specific experiment
          'user_latency': 0         # Set the values which correspond to real life participation delay for users 
    
         }




# Following class is currently not used

class ConsumptionModel(object):
    def __init__(self, df, params):
        self.df = df
        self.params = params

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
        if self.params["season"] == -1:
            month = random.randrange(1,12)
        elif self.params["season"] == 0:
            month = random.choice(self.spring)
        elif self.params["season"] == 1:
            month = random.choice(self.summer)
        elif self.params["season"] == 2:
            month = random.choice(self.autumn)
        elif self.params["season"] == 3:
            month = random.choice(self.winter)
            
#         Select the day of week
        if self.params["day_of_week"] == -1:
#             Select random day
            dow = random.randrange(0,7)
        else:
            dow = self.params["day_of_week"] 
            
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
        self.noisy_tariff["h1_start"] = [random.choice(range(constraints["h1_start"]-2, constraints["h1_start"]+3)) for _ in range(len(self.avail_users))]
        self.noisy_tariff["h1_end"] = [random.choice(range(constraints["h1_end"]-4, constraints["h1_end"]+2)) for _ in range(len(self.avail_users))]
    

    def tariff_policy(self, levels, constraints):
#         use variables from auto_noise_addition and input variables of this function to create a tariff policy 
#         for each participating user **Needs more attention
        self.auto_noise_addition(levels,constraints)
    
        d = np.ones((48, len(self.avail_users)))
        self.df_tariff = pd.DataFrame(data=d, columns = self.avail_users)
        for i in range(len(self.avail_users)):
            self.df_tariff.loc[self.noisy_tariff["h1_start"][i]:self.noisy_tariff["h1_end"][i], self.avail_users[i]] = 2

        self.df_tariff.index = self.sample.index
        
        
    def run(self):
#         FOR EACH USER, call test function of consumption model, get modified behaviour, return original data point and modified data point
        self.sample = self.sample.interpolate(method = 'linear', axis = 0).ffill().bfill()
        self.sample = self.sample.join(self.df_weather.loc[self.sample.index,:])
        self.df_response = pd.DataFrame()
        self.sample["hour"] = self.sample.index.hour
        self.sample["day"] = self.sample.index.day
        self.sample["month"] = self.sample.index.month

        for i in range(len(self.avail_users)):
            one_hot= pd.get_dummies(self.df_tariff[self.avail_users[i]])
            one_hot_renamed = one_hot.rename(index=str, columns={1.0:'normal', 2.0:'high', 3.0:'low'}) 
            self.sample = pd.concat([self.sample, one_hot_renamed], axis =1)
            self.sample["low"] = 0

            self.sample["trial_n"] = self.sample[self.avail_users[i]]
            
            self.test()
            self.df_response[self.avail_users[i]] = self.preds
            self.sample = self.sample.drop(['low', 'normal', 'high', 'trial_n'], axis= 1)
            
        self.df_response['mean']= df_response.mean(axis = 1)
        return df_response['mean']
            
            
            
    def test(self):
        self.preds = self.sample['trial_n']
        self.preds.loc[self.sample['high']==1] = self.preds.loc[self.sample['high']==1]*(8/(self.params['active_users']
                                                                                            *self.params['avail_users']))
        
        