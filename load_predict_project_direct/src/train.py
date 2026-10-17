import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib

plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['font.size']=15

class PowerLoadModel:
    def __init__(self,file_path):
        logfile_name='train'+pd.to_datetime(datetime.now()).strftime('%Y%m%d%H%M%S')

        self.logfile=Logger(logfile_name).get_logger()

        self.logfile.info('开始创建 电力负荷模型 对象了')

        self.data_source=data_preprocessing(file_path)

def ana_data(data):
    data_copy=data.copy()

    fig=plt.figure(figsize=(40,40))
    ax1=fig.add_subplot(221)
    ax1.hist(data_copy['power_load'],bins=100)
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')

    data_copy['hour']=data_copy['time'].dt.hour
    # print(data_copy.head())
    hour_load_mean=data_copy.groupby('hour',as_index=False)['power_load'].mean()
    # print(hour_load_mean)

    ax2=fig.add_subplot(222)
    ax2.plot(hour_load_mean['hour'],hour_load_mean['power_load'])
    ax2.set_title('各个小时的平均负荷趋势')
    ax2.set_xticks(range(0,24))
    ax2.set_xticklabels(range(0,24),rotation=45)
    ax2.set_xlabel('小时')

    data_copy['month']=data_copy['time'].dt.month
    month_load_mean=data_copy.groupby('month',as_index=False)['power_load'].mean()
    ax3=fig.add_subplot(223)
    ax3.plot(month_load_mean['month'],month_load_mean['power_load'])
    ax3.set_title('各个月份的平均负荷趋势')
    ax3.set_xticks(range(1,13)) 
    ax3.set_xlabel('月份')

    data_copy['weekday']=data_copy['time'].dt.weekday+1
    data_copy['is_workday']=(data_copy['weekday']<=5).astype(int)
    day_load_mean=data_copy.groupby('is_workday',as_index=False)['power_load'].mean()
    print(day_load_mean)
    ax4=fig.add_subplot(224)
    ax4.bar(['周末','工作日'],day_load_mean['power_load'])
    ax4.set_title('工作日和周末的平均平均负荷')
 
    # plt.savefig(r'data/fig/负荷整体的分布情况.png')
    # plt.show()

    # print(data_copy.head())

    # return data_copy
#特征工程
def feature_engineering(data,logger):
    feature_data=data.copy()
    feature_data['month']=feature_data['time'].dt.month
    feature_data['hour']=feature_data['time'].dt.hour

    feature_data=pd.get_dummies(
        feature_data,
        columns=['month', 'hour'],
        dtype=int)

    feature_data['current_load']=feature_data['power_load']
    
    feature_data['lag_1_load']=feature_data['power_load'].shift(1)
    feature_data['lag_2_load']=feature_data['power_load'].shift(2)
    feature_data['lag_3_load']=feature_data['power_load'].shift(3)

    yesterday_time=feature_data['time']-pd.to_timedelta('1d')

    time_load_dict=feature_data.set_index('time')['power_load'].to_dict()

    feature_data['yesterday_load']=yesterday_time.apply(lambda x: time_load_dict.get(x))

    feature_data=feature_data.dropna()

    return feature_data
#模型训练
def model_train(data,feature,logger):

    split_index=int(len(data)*0.8)
    train_data=data.iloc[:split_index].copy()
    test_data=data.iloc[split_index:].copy()

    train_h=train_data.copy()
    test_h=test_data.copy()

    for h in [1,2,3]:
        train_h[f'target_h{h}']=train_data['power_load'].shift(-h)
        test_h[f'target_h{h}']=test_data['power_load'].shift(-h)

    logger.info('---网格搜索 + 交叉验证 寻找最优超参数---')
    logger.info(f'开始时间:{datetime.now()}')

    param_dict={
        'n_estimators':[80,100,150,200],
        'max_depth':[2,4,6,8],
        'learning_rate':[0.01,0.05,0.1]
    }


    for h in range(1,4):
        train_cureent=train_h.dropna(subset=[f'target_h{h}'])
        test_current=test_h.dropna(subset=[f'target_h{h}'])

        target_col=f'target_h{h}'

        target_col_all=['target_h1','target_h2','target_h3']

        x_train=train_cureent.drop(columns=['power_load','time',*target_col_all])
        y_train=train_cureent[target_col]

        print(x_train.columns.tolist())

        x_test=test_current.drop(columns=['power_load','time',*target_col_all])
        y_test=test_current[target_col]

        cv=TimeSeriesSplit(n_splits=5,gap=h)

        estimator=XGBRegressor()

        xgb_model=GridSearchCV(estimator=estimator,param_grid=param_dict,cv=cv,scoring='r2')

        xgb_model.fit(x_train,y_train)

        logger.info(f'最优参数组合:{xgb_model.best_params_}')
        logger.info(f'最优参数组合的R^2:{xgb_model.best_score_}')

        best_model=xgb_model.best_estimator_
        y_pred=best_model.predict(x_test)
        logger.info(f'测试集MAE:{mean_absolute_error(y_test,y_pred)}')
        logger.info(f'测试集MSE:{mean_squared_error(y_test,y_pred)}')
        logger.info(f'测试集R^2:{r2_score(y_test,y_pred)}')
        logger.info(f'测试集平均负荷: {y_test.mean():.4f}')

        y_pred_baseline=x_test['current_load']
        logger.info(f'小时基线测试集MAE:{mean_absolute_error(y_test,y_pred_baseline)}')
        logger.info(f'小时基线测试集MSE:{mean_squared_error(y_test,y_pred_baseline)}')
        logger.info(f'小时基线测试集R^2:{r2_score(y_test,y_pred_baseline)}')

        y_pred_baseline_day=x_test['yesterday_load']
        logger.info(f'日基线测试集MAE:{mean_absolute_error(y_test,y_pred_baseline_day)}')
        logger.info(f'日基线测试集MSE:{mean_squared_error(y_test,y_pred_baseline_day)}')
        logger.info(f'日基线测试集R^2:{r2_score(y_test,y_pred_baseline_day)}')

        filename=f'xgb_direct_{h}_20261015.pkl'

        model_dir=Path('./model')

        model_dir.mkdir(parents=True,exist_ok=True)

        save_path=model_dir/filename

        joblib.dump(best_model,filename=save_path)
        logger.info(f'模型{h}保存成功')

    logger.info(f'结束时间:{datetime.now()}')



if __name__ == '__main__':
    pm=PowerLoadModel('train.csv')
    # print(pm.data_source)
    # ana_data(pm.data_source)
    feature_data=feature_engineering(pm.data_source,pm.logfile)
    model_train(feature_data,feature_data.columns,pm.logfile)


