import simpy
import numpy as np


class QueuingSystemModel:
    def __init__(self,
                 channels_number,
                 service_flow_rate,
                 queue_waiting_flow_rate=None,
                 ):
        self.env = simpy.Environment()
        self.channel = simpy.Resource(self.env, channels_number)

        self.service_flow_rate = service_flow_rate
        self.queue_waiting_flow_rate = queue_waiting_flow_rate

        self.applications_in_queue_time = []
        self.applications_in_queue = []
        self.applications_QS_times = []
        self.rejected_applications = []
        self.total_applications = []
        self.processed_applications = []

    def __application_processing(self):
        yield self.env.timeout(np.random.exponential(1 / self.service_flow_rate))

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

    def __simulate_process(self, max_queue_length, channels_number):
        self.total_applications.append(len(self.channel.queue) + self.channel.count)
        self.applications_in_queue.append(len(self.channel.queue))

        with self.channel.request() as request:
            if len(self.channel.queue) <= max_queue_length:
                start_time = self.env.now
                self.processed_applications.append(len(self.channel.queue) + self.channel.count)
                application_to_processing = yield request | self.env.process(self.__application_waiting())
                self.applications_in_queue_time.append(self.env.now - start_time)

                if request in application_to_processing:
                    yield self.env.process(self.__application_processing())

                self.applications_QS_times.append(self.env.now - start_time)
            else:
                self.rejected_applications.append(channels_number + max_queue_length + 1)
                self.applications_in_queue_time.append(0)
                self.applications_QS_times.append(0)

    def __run_simulation(self, max_queue_length, applications_flow_rate, channels_number):
        while True:
            yield self.env.timeout(np.random.exponential(1 / applications_flow_rate))
            self.env.process(self.__simulate_process(max_queue_length, channels_number))

    def run(self, run_count, max_queue_length, applications_flow_rate, channels_number):
        self.env.process(self.__run_simulation(max_queue_length, applications_flow_rate, channels_number))
        self.env.run(until=run_count)
