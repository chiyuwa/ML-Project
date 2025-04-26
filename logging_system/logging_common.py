import logging   # 日志模块
import datetime   # 时间模块
import time
import os

PATH_LOGGING = r'./Log/'
# PATH_LOGGING = r'/media/microport/StorageDevice/zez/aimodule/log_ufa/'


def timestamp2time(timestamp):
    timeStruct = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)
def get_file_created_time(file_path): # '''获取文件的创建时间'''
    # file_path = unicode(file_path,'utf8')
    t = os.path.getctime(file_path)
    return timestamp2time(t)
def get_file_modified_time(file_path): # '''获取文件的修改时间'''
    # file_path = unicode(file_path, 'utf8')
    t = os.path.getmtime(file_path)
    return timestamp2time(t)
def get_file_access_time(file_path): # '''获取文件的访问时间'''
    # file_path = unicode(file_path, 'utf8')
    t = os.path.getatime(file_path)
    return timestamp2time(t)
def get_file_size(file_path): # '''获取文件的大小,结果保留两位小数，单位为MB'''
    # file_path = unicode(file_path,'utf8')
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)

# 清理上个月的日志
def clean_loggers(path=PATH_LOGGING):
    today_date = str(datetime.date.today()) # 获取今天的日期 格式2019-08-01

    # 遍历目录下的所有日志文件 i是文件名
    for filename in os.listdir(path):
        file_path = path + filename    # 生成日志文件的路径
        t = get_file_modified_time(file_path)
        # 获取日志的年月，和今天的年月
        m, today_m = int(t[5:7]), int(today_date[5:7])   # 日志，今天的月份
        y, today_y = int(t[0:4]), int(today_date[0:4])   # 日志，今天的年份
        # 对上个月的日志进行清理，即删除
        if y <= today_y and m < today_m:
            if os.path.exists(file_path):   # 判断生成的路径对不对，防止报错
                os.remove(file_path)   # 删除文件

def get_logger(name, path=PATH_LOGGING):
    # 设置日志存放路径
    if not os.path.exists(PATH_LOGGING): os.mkdir(PATH_LOGGING)
    logger = logging.getLogger(name)
    filename = f'{datetime.datetime.now().date()}_{name}.log'
    
    if not logger.handlers:
        # 文件处理器
        fh = logging.FileHandler(path+filename, mode='a+', encoding='utf-8')
        # 控制台处理器（新增部分）
        ch = logging.StreamHandler()
        
        # 统一格式
        log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        formatter = logging.Formatter(log_format)
        
        logger.setLevel(logging.DEBUG)
        
        # 设置处理器格式
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)  # 控制台使用相同格式
        
        # 添加处理器
        logger.addHandler(fh)
        logger.addHandler(ch)       # 新增控制台处理器

    return logger


def unittest_logging():
    logger1 = get_logger('m1')
    logger2 = get_logger('m2')

    logger1.info("info")
    logger2.warning("warning")


if __name__ == '__main__':
    unittest_logging()
    # clean_loggers()