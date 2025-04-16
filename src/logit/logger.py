import sys
from datetime import datetime

sys.path.append('src')
from logit.config import *
from logit.errors import *
from logit.utils import *


_levelToName = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    NOTSET: 'NOTSET',
}
_nameToLevel = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET,
}


class Logger:
    '''
    Logger with output to console and file (optional).

    `name`:             Name of the logger
    `level`:            Minimum log level (NOTSET, DEBUG, INFO, WARNING,
                        ERROR, CRITICAL)
    `datefmt`:          Date format for the timestamp
    `log_file_path`:    The path to the log file. If None, logs will only
                        be printed to the console.
    '''
    def __init__(self,
                 name: str = 'Logger',
                 level: str = 'INFO',
                 datefmt: str = '%Y-%m-%d %H:%M:%S',
                 log_file_path: str | None = None) -> None:
        
        level = level.upper()

        self.levels = _nameToLevel.copy()
        
        if level not in self.levels: raise InvalidLogLevelError(level)

        self.name = name
        self.level = level
        self.log_file_path = log_file_path
        self.datefmt = datefmt

    def _get_timestamp(self) -> str:
        return datetime.now().strftime(self.datefmt)
    
    def _get_log_level(self,
                       level: str) -> int:
        
        return self.levels[level.upper()]
    
    def _log(self,
             level: str,
             message: str) -> None:

        if self._get_log_level(level) >= self._get_log_level(self.level):
            timestamp = self._get_timestamp()
            log_message = f'[{timestamp}] [{level}] {message}'
            
            sys.stdout.write(log_message + '\n')
            sys.stdout.flush()
            
            if self.log_file_path:
                write_to_file(self.log_file_path, log_message)

    def set_level(self,
                  level: str) -> None:
        
        level = level.upper()
        if level in self.levels: self.level = self.levels[level]
        else: raise InvalidLogLevelError(level)

    def debug(self,
              message: str) -> None:
        
        self._log('DEBUG', message)

    def info(self,
              message: str) -> None:
        
        self._log('INFO', message)

    def warning(self,
              message: str) -> None:
        
        self._log('WARNING', message)

    def error(self,
              message: str) -> None:
        
        self._log('ERROR', message)

    def critical(self,
              message: str) -> None:
        
        self._log('CRITICAL', message)

    def blank(self) -> None:
        
        sys.stdout.write('\n')
        sys.stdout.flush()
            
        if self.log_file_path:
            write_to_file(self.log_file_path, '')
