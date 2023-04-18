
import time
from typing import Any, Tuple, Callable


class StopWatch:
    def __init__(self,
                 on_end: Callable = None,
                 args: Tuple[Any, ...] = (),
                 ) -> None:
        
        self.args = args
        self.on_end = on_end
        self.stime = None
        self.etime = None
        self.process_time = None
    
    def __enter__(self):
        self.stime = time.time()
        
    def __exit__(self, type, value, traceback):
        self.etime = time.time()
        self.process_time = self.etime - self.stime
        if self.on_end is not None:
            self.on_end(self.process_time,
                        *self.args)