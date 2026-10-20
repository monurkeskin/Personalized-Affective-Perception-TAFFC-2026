from CLModel.cl_model_c3_faster import get_arousal_valence
from threading import Thread
import time
from timeit import default_timer as timer
from datetime import timedelta


SLEEPING = 0.25


class TrainManager:
    queue: list
    main_thread: Thread
    __training: bool
    __runining: bool

    def __init__(self, sess, graph):
        """
            Constructor starts the thread
        :param sess: Gloabal TF Session Object
        :param graph: Global TF Graph Object
        """
        self.queue = []
        self.main_thread = Thread(target=self.__run_main)
        self.__training = False
        self.__runining = True

        self.sess = sess
        self.graph = graph

        self.main_thread.start()

    def __run_main(self):
        """
            Queue Control Thread
        :return: None
        """
        while self.__runining:
            if self.__training == True or len(self.queue) == 0:
                time.sleep(SLEEPING)

                continue

            self.__run_training()

    def __run_training(self):
        """
            Training Thread
        :return: None
        """
        self.__training = True
        round_dir, gwr_dir, round = self.queue.pop(0)  # Dequeue

        print("Training Round: ", round)
        start_time = timer()
        get_arousal_valence(round_dir, gwr_dir, self.graph, self.sess)
        end_time = timer()

        print(
            "Elapsed Time (Round: %d):" % round,
            timedelta(seconds=end_time - start_time),
        )

        self.__training = False

    def enqueue(self, round_dir: str, gwr_dir: str, round: int):
        """
            Enqueue training.
        :param round_dir: round_dir from Session object.
        :param gwr_dir: gwr_dir from Session object
        :param round: Training round
        :return: None
        """
        self.queue.append([round_dir, gwr_dir, round])

    def stop(self):
        """
            Close Train Manager
        :return: None
        """
        if self.__runining and self.main_thread.is_alive():
            self.__runining = False
            self.main_thread.join()

    def completed(self) -> bool:
        """
            Check queue is empty or not
        :return: Bool
        """
        return not self.__training and len(self.queue) == 0

    def wait(self):
        """
            Use this method to clean queue
        :return: None
        """
        while not self.completed():
            time.sleep(SLEEPING)

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return self.queue.__str__()

    def __repr__(self):
        return self.queue.__repr__()

    def __hash__(self):
        return self.queue.__hash__()
