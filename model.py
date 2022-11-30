import pandas as pd
from datetime import timedelta
from sklearn.metrics import  accuracy_score,mean_squared_error,recall_score,f1_score,precision_score
from velov import utils

station_info = utils.get_stations_info()
last_week_available_stations = [u'velov-10001', u'velov-10002', u'velov-10004', u'velov-10005', u'velov-10006', u'velov-10007',
u'velov-10008', u'velov-1001', u'velov-10011', u'velov-10012', u'velov-10013', u'velov-10014', u'velov-10016', u'velov-10018',
u'velov-1002', u'velov-10021', u'velov-10024', u'velov-10025', u'velov-10027', u'velov-10028', u'velov-1003', u'velov-10030',
u'velov-10031', u'velov-10032', u'velov-10034', u'velov-10035', u'velov-10036', u'velov-10038', u'velov-10039', u'velov-10041',
u'velov-10043', u'velov-10046', u'velov-10047', u'velov-10048', u'velov-10049', u'velov-1005', u'velov-10053', u'velov-10054',
u'velov-10055', u'velov-10056', u'velov-10058', u'velov-10059', u'velov-1006', u'velov-10060', u'velov-10061', u'velov-10063',
u'velov-10064', u'velov-10071', u'velov-10072', u'velov-10073', u'velov-10074', u'velov-10075', u'velov-10079', u'velov-10080',
u'velov-10083', u'velov-10084', u'velov-10086', u'velov-10087', u'velov-10088', u'velov-10091', u'velov-10092', u'velov-10101',
u'velov-10102', u'velov-10103', u'velov-10110', u'velov-10111', u'velov-10112', u'velov-10113', u'velov-10114', u'velov-10115',
u'velov-10116', u'velov-10117', u'velov-10118', u'velov-10119', u'velov-1012', u'velov-10120', u'velov-10121', u'velov-10122',
u'velov-1013', u'velov-1016', u'velov-1020', u'velov-1021', u'velov-1022', u'velov-1023', u'velov-1024', u'velov-1031', u'velov-1032',
u'velov-1034', u'velov-1035', u'velov-1036', u'velov-11001', u'velov-11002', u'velov-11003', u'velov-12001', u'velov-12002', u'velov-2001',
u'velov-2002', u'velov-2003', u'velov-2004', u'velov-2005', u'velov-2006', u'velov-2007', u'velov-2008', u'velov-2009', u'velov-2010',
u'velov-2011', u'velov-2012', u'velov-2013', u'velov-2014', u'velov-2015', u'velov-2016', u'velov-2017', u'velov-2020', u'velov-2022',
u'velov-2023', u'velov-2024', u'velov-2025', u'velov-2026', u'velov-2028', u'velov-2030', u'velov-2035', u'velov-2036', u'velov-2037',
u'velov-2038', u'velov-2039', u'velov-2041', u'velov-2042', u'velov-3001', u'velov-3002', u'velov-3003', u'velov-3004', u'velov-3005',
u'velov-3007', u'velov-3008', u'velov-3009', u'velov-3010', u'velov-3012', u'velov-3013', u'velov-3015', u'velov-3016', u'velov-3018',
u'velov-3021', u'velov-3024', u'velov-3029', u'velov-3031', u'velov-3032', u'velov-3036', u'velov-3037', u'velov-3038', u'velov-3039',
u'velov-3043', u'velov-3044', u'velov-3051', u'velov-3053', u'velov-3058', u'velov-3066', u'velov-3067', u'velov-3071', u'velov-3079',
u'velov-3080', u'velov-3082', u'velov-3083', u'velov-3084', u'velov-3085', u'velov-3086', u'velov-3087', u'velov-3088', u'velov-3089',
u'velov-3090', u'velov-3091', u'velov-3094', u'velov-3097', u'velov-3099', u'velov-3100', u'velov-3101', u'velov-3102', u'velov-3103',
u'velov-4001', u'velov-4002', u'velov-4003', u'velov-4004', u'velov-4005', u'velov-4006', u'velov-4007', u'velov-4009', u'velov-4011',
u'velov-4012', u'velov-4014', u'velov-4017', u'velov-4021', u'velov-4022', u'velov-4023', u'velov-4024', u'velov-4025', u'velov-4026',
u'velov-4041', u'velov-4042', u'velov-5001', u'velov-5002', u'velov-5004', u'velov-5005', u'velov-5006', u'velov-5007', u'velov-5008',
u'velov-5009', u'velov-5015', u'velov-5016', u'velov-5026', u'velov-5029', u'velov-5030', u'velov-5031', u'velov-5036', u'velov-5040',
u'velov-5041', u'velov-5044', u'velov-5045', u'velov-5047', u'velov-5050', u'velov-5053', u'velov-5054', u'velov-5055', u'velov-6001',
u'velov-6002', u'velov-6003', u'velov-6004', u'velov-6005', u'velov-6006', u'velov-6007', u'velov-6008', u'velov-6011', u'velov-6012',
u'velov-6016', u'velov-6020', u'velov-6021', u'velov-6022', u'velov-6023', u'velov-6024', u'velov-6028', u'velov-6031', u'velov-6032',
u'velov-6035', u'velov-6036', u'velov-6037', u'velov-6040', u'velov-6041', u'velov-6042', u'velov-6043', u'velov-6044', u'velov-6045',
u'velov-7001', u'velov-7002', u'velov-7003', u'velov-7004', u'velov-7005', u'velov-7006', u'velov-7007', u'velov-7008', u'velov-7009',
u'velov-7010', u'velov-7011', u'velov-7012', u'velov-7013', u'velov-7014', u'velov-7016', u'velov-7018', u'velov-7020', u'velov-7021',
u'velov-7022', u'velov-7023', u'velov-7024', u'velov-7030', u'velov-7031', u'velov-7033', u'velov-7034', u'velov-7035', u'velov-7038',
u'velov-7039', u'velov-7041', u'velov-7045', u'velov-7046', u'velov-7049', u'velov-7052', u'velov-7053', u'velov-7055', u'velov-7056',
u'velov-7057', u'velov-7061', u'velov-7062', u'velov-7064', u'velov-8001', u'velov-8002', u'velov-8003', u'velov-8004', u'velov-8006',
u'velov-8007', u'velov-8008', u'velov-8009', u'velov-8010', u'velov-8011', u'velov-8015', u'velov-8020', u'velov-8021', u'velov-8024',
u'velov-8025', u'velov-8029', u'velov-8030', u'velov-8034', u'velov-8035', u'velov-8037', u'velov-8038', u'velov-8039', u'velov-8040',
u'velov-8041', u'velov-8042', u'velov-8051', u'velov-8052', u'velov-8053', u'velov-8054', u'velov-8056', u'velov-8057', u'velov-8058',
u'velov-8059', u'velov-8060', u'velov-8061', u'velov-9002', u'velov-9003', u'velov-9004', u'velov-9006', u'velov-9008', u'velov-9010',
u'velov-9011', u'velov-9013', u'velov-9014', u'velov-9020', u'velov-9022', u'velov-9029', u'velov-9032', u'velov-9033', u'velov-9040',
u'velov-9041', u'velov-9042', u'velov-9043', u'velov-9044', u'velov-9049', u'velov-9050', u'velov-9051', u'velov-9052']



def classify(bikes : int,bike_stands : int, capacity : int,tolerance_level = 0.1) -> int:
    '''
    input : number of available bikes and bike stands, station capacity and tolerance level to deem a station nearly empty or nearly full
    ---
    result : number indicating station status:
        -  0 - empty - there is no bike available;
        -  1 - nearly empty - there is less than a percentage based on the tolerance level of the bikes available;
        -  2 - normal - there is both a satisfying number of bikes and bike stands at the station;
        -  3 - nearly full - there is less than a percentage based on the tolerance level of the bike stands available;
        -  4 - full - the station is full, there is no bike stand available.
    '''
    if bikes <= 0:
        return 0
    if bikes < tolerance_level * capacity:
        return 1
    if bike_stands <=0:
        return 4
    if bike_stands < tolerance_level * capacity:
        return 3
    return 2

def classify_station(station_data : pd.DataFrame) -> pd.DataFrame:
    classified_station_data = station_data.copy()
    classified_station_data['status_code']= classified_station_data.apply(lambda x: classify(x.bikes,x.bike_stands,utils.get_stations_info().capacity.values[0]),axis = 1)
    classified_station_data['status'] = classified_station_data['status_code'].map({0:"empty",1:"nearly empty",2:"OK",3:"nearly full",4:'full'})
    classified_station_data['is_empty'] = classified_station_data['status_code'].map({0:1,1:0,2:0,3:0,4:0})
    classified_station_data['is_full'] = classified_station_data['status_code'].map({0:0,1:0,2:0,3:0,4:1})
    
    return classified_station_data


def predict(station_data : pd.DataFrame,n_min = 15,n_days = 7) -> pd.DataFrame:
    '''
    input : station_data (classified), number of minutes (multiple of 5) of offset, with the number of days of data considered.
    ---
    return : baseline model prediction
    '''
    y = station_data.copy().reset_index().iloc[-n_days*24*12:]
    y_pred = station_data.copy().reset_index().iloc[int(-(n_days*24*12+n_min/5)):-int(n_min/5)]
    y_pred['time']=y_pred['time'] + timedelta(minutes = n_min)
    y_pred = y_pred.reset_index().drop(columns = 'index')
    return y,y_pred

def compute_metrics(classified_station_data,n_days = 7):
    mse = []
    precision_empty = []
    precision_full = []  
    accuracy = []
    step =[]
    
    steps = range(1,13)
    
    for i in steps:
        y,y_pred = predict(classified_station_data,n_min = i*5,n_days=n_days)
        mse.append(mean_squared_error(y.bikes,y_pred.bikes))
        precision_empty.append(precision_score(y.is_empty,y_pred.is_empty))
        precision_full.append(precision_score(y.is_full,y_pred.is_full))
        accuracy.append(accuracy_score(y.status_code,y_pred.status_code))
        step.append(f'{str(int((i*5)))} m')    

  
    result=pd.DataFrame(index=['step','mse','precision_empty','precision_full','accuracy'],data=[step,mse,precision_empty,precision_full,accuracy]).transpose()
       
    return result
    