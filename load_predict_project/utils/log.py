# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self,  log_name, level='info', fmt='%(asctime)s - %(levelname)s: %(message)s'):
        # 指定日志保存的路径
      

        # 初始logger名称和格式
        self.log_name = log_name

        # 初始格式
        self.fmt = fmt

        # 先声明一个 Logger 对象
        self.logger = logging.getLogger(log_name)

        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))

    def get_logger(self):
        # 指定对应的 Handler 为 FileHandler 对象， 这个可适用于多线程情况

        # 当前文件：.../load_predict_project/utils/log.py
        current_file = Path(__file__).resolve()

        # 项目根目录：.../load_predict_project
        root_path = current_file.parent.parent

        # 日志目录：.../load_predict_project/log
        log_path = root_path / 'log'
        log_path.mkdir(parents=True, exist_ok=True)

        # 日志文件
        file_name = log_path / f'{self.log_name}.log'
        
        rotate_handler = logging.FileHandler(file_name, encoding="utf-8", mode="a")

        # Handler 对象 rotate_handler 的输出格式
        formatter = logging.Formatter(self.fmt)
        rotate_handler.setFormatter(formatter)

        # 将rotate_handler添加到Logger
        self.logger.addHandler(rotate_handler)

        return self.logger

#测试日志功能
if __name__=='__main__':
    
    # logger=Logger('hg_test').get_logger()

    # logger.info('我是普通日志信息')
    # logger.error('我是错误日志信息')

    # try:
    #     logger.info('开始计算')
    #     print(10/0)
    # except Exception as e:
    #     logger.error(f'计算出错，原因是{e}')
    # else:
    #     logger.info('计算成功')
    # finally:
    #     logger.info('计算解释')

    new_name='train'+pd.to_datetime(datetime.now()).strftime('%Y%m%d%H%M%S')

    logger=Logger(new_name).get_logger()

    try:
        logger.info('开始计算')
        print(10/0)
    except Exception as e:
        logger.error(f'计算出错，原因是{e}')
    else:
        logger.info('计算成功')
    finally:
        logger.info('计算解释')

