class InvalidLogLevelError(Exception):
    
    def __init__(self,
                 level: str) -> None:
        
        super().__init__(f"Invalid log level: '{level}'. Valid levels are: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL.")
        self.level = level
