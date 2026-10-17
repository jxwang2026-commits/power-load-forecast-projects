import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.ticker as mick
import matplotlib.dates as mdates
import joblib

plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size']=15

class PowerLoadPredict:
    def __init__(self,file_path):
        logfile_name='predict'+pd.to_datetime(datetime.now()).strftime('%Y%m%d%H%M%S')

        self.logger=Logger(logfile_name).get_logger()

        self.data_source=data_preprocessing(file_path)

        self.time_load_dict=self.data_source.set_index('time').power_load.to_dict()

def pred_feature_extract(data_dict,time,logger):
    # logger.info(f'======解析预测时间：{time}所对应的特征======')

    hour_feature=[f'hour_{i}' for i in range(0,24)]
    month_feature=[f'month_{i}' for i in range(1,13)]
    lag_feature=['lag_1_load', 'lag_2_load', 'lag_3_load', 'yesterday_load']
    feature_names=month_feature+hour_feature+lag_feature

    # hour_list=[]
    # pred_hour=time.hour
    # for i in range(24):
    #     if pred_hour==i:
    #         hour_list.append(1)
    #     else:
    #         hour_list.append(0)
    hour_list=[0]*24
    hour_list[time.hour]=1

    # month_list=[]
    # pred_month=time.month
    # for i in range(1,13):
    #     if pred_month==i:
    #         month_list.append(1)
    #     else:
    #         month_list.append(0)
    month_list=[0]*12
    month_list[time.month-1]=1

    lag_1_time=time-pd.Timedelta(hours=1)
    lag_1_load=data_dict.get(lag_1_time)
    lag_2_time=time-pd.Timedelta(hours=2)
    lag_2_load=data_dict.get(lag_2_time)
    lag_3_time=time-pd.Timedelta(hours=3)
    lag_3_load=data_dict.get(lag_3_time)

    yesterday_time=time-pd.Timedelta(days=1)
    yesterday_load=data_dict.get(yesterday_time)

    lag_values = [
    lag_1_load,
    lag_2_load,
    lag_3_load,
    yesterday_load
    ]

    if any(value is None
    for value in lag_values):
        raise ValueError(
        f'{time}历史特征不足'
        )

    feature_data=month_list+hour_list+[lag_1_load,lag_2_load,lag_3_load,yesterday_load]

    feature_df=pd.DataFrame([feature_data],columns=feature_names)

    return feature_df

def predict_plot(data):
    fig=plt.figure(figsize=(100,100))
    ax=fig.add_subplot(111)
    ax.plot(data['预测时间'],data['真实负荷'],color='blue',label='真实负荷')
    ax.plot(data['预测时间'],data['预测负荷'],color='yellow',label='预测负荷')
    # ax.plot(data['预测时间'],data['误差'],color='red',label='误差')
    ax.set_title('时间与真实负荷和预测负荷关系图',fontsize=30)
    ax.set_xlabel('预测时间')
    ax.set_ylabel('负荷')
    ax.grid(True,linestyle='--',alpha=0.5)
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=50))
    ax.tick_params(axis='x', rotation=45)
    plt.savefig(r'data\fig\真实负荷和预测负荷关系图.png')
    plt.show()

if __name__ == '__main__':
    power_load_predict=PowerLoadPredict('test.csv')

    estimator=joblib.load(r'model\xbg_20261013.pkl')

    predict_times=power_load_predict.data_source['time'][power_load_predict.data_source['time']>=pd.to_datetime('2015-08-01 00:00:00')]

    evaluate_list=[]

    power_load_predict.logger.info('开始预测')

    for pre_times in predict_times:

        time_load_dict_masked={k:v for k,v in power_load_predict.time_load_dict.items() if k < pre_times}

        # print(f'预测的时间:{pre_times}')

        feature_df=pred_feature_extract(time_load_dict_masked,pre_times,power_load_predict.logger)

        y_pred=estimator.predict(feature_df)[0]

        true_load=power_load_predict.time_load_dict.get(pre_times)

        if true_load is None:
            power_load_predict.logger.warning(f'{pre_times}没有真实负荷,跳过评价')
            continue

        error=true_load-y_pred

        evaluate_list.append([pre_times,true_load,y_pred,error])

    evaluate_df=pd.DataFrame(
        evaluate_list,columns=['预测时间','真实负荷','预测负荷','误差']
    )

    power_load_predict.logger.info(evaluate_df)

    power_load_predict.logger.info(f'MAE:{mean_absolute_error(evaluate_df['真实负荷'],evaluate_df['预测负荷'])}')
    power_load_predict.logger.info(f'MSE:{mean_squared_error(evaluate_df['真实负荷'],evaluate_df['预测负荷'])}')
    power_load_predict.logger.info(f'R^2:{r2_score(evaluate_df['真实负荷'],evaluate_df['预测负荷'])}')

    power_load_predict.logger.info('结束预测')

    predict_plot(evaluate_df)

    