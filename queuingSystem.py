import simpy
import numpy as np


class QueuingSystem:
    def __init__(self,
                 channels_number,
                 processing_flow_rate,
                 queue_waiting_flow_rate,
                 applications_flow_rate,
                 ):
        self.env = simpy.Environment()
        self.channel = simpy.Resource(self.env, channels_number)
        self.channels_number = channels_number

        self.processing_flow_rate = processing_flow_rate
        self.queue_waiting_flow_rate = queue_waiting_flow_rate
        self.applications_flow_rate = applications_flow_rate

        self.applications_in_queue_time = []
        self.applications_QS_times = []
        self.applications_in_queue = []
        self.rejected_applications = []
        self.total_applications = []
        self.processed_applications = []

    def __application_processing(self):
        yield self.env.timeout(np.random.exponential(1 / self.processing_flow_rate))

    def __application_waiting(self):
        yield self.env.timeout(np.random.exponential(1 / self.queue_waiting_flow_rate))

    def __process_application(self, request):
        start_time = self.env.now
        self.processed_applications.append(len(self.channel.queue) + self.channel.count)
        res = yield request | self.env.process(self.__application_waiting())
        self.applications_in_queue_time.append(self.env.now - start_time)

        if request in res:
            yield self.env.process(self.__application_processing())

        self.applications_QS_times.append(self.env.now - start_time)

    def __simulate_process(self):
        self.total_applications.append(len(self.channel.queue) + self.channel.count)
        self.applications_in_queue.append(len(self.channel.queue))

        with self.channel.request() as request:
            current_queue_len = len(self.channel.queue)
            current_channel_count = self.channel.count

            start_time = self.env.now
            self.processed_applications.append(current_queue_len + current_channel_count)

            yield request
            self.applications_in_queue_time.append(self.env.now - start_time)
            yield self.env.process(self.__application_processing())

            self.applications_QS_times.append(self.env.now - start_time)

    def __run_simulation(self):
        application = 0

        while True:
            yield self.env.timeout(np.random.exponential(1 / self.applications_flow_rate))
            self.env.process(self.__simulate_process())
            application += 1

    def run(self, run_count):
        self.env.process(self.__run_simulation())
        self.env.run(until=run_count)
