import logging
import os

class LogTool:
    def __init__(self, log_file="app.log", log_level=logging.INFO):
        """
        初始化日志工具。
        
        :param log_file: 日志文件名
        :param log_level: 日志级别，例如 logging.INFO、logging.DEBUG
        """
        self.logger = logging.getLogger("LogTool")
        self.logger.setLevel(log_level)

        # 创建日志文件目录（如果不存在）
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 日志格式
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 文件日志处理器
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        获取日志对象。
        """
        return self.logger

