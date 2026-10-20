import time


class NegotiationTimeController:
    def __init__(self, deadline):
        self.__deadline = deadline
        self.__current_time = 0
        # TESTING PURPOSES.
        self.index = 1

    def start_negotiation_time(self):
        self.__start_time = time.time()

    def time_is_up(self):
        self.__current_time = time.time() - self.__start_time
        if self.__current_time <= self.__deadline:
            return False
        else:
            return True

    def get_deadline(self):
        return self.__deadline

    def get_current_time(self):
        self.__current_time = time.time() - self.__start_time
        return float(self.__current_time) / self.__deadline * 1.0

    # def get_current_time(self):
    #     discretize = (self.index * 1.0) / 20
    #     return discretize
    #
    # def increase_index(self):
    #     self.index += 1
