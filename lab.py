from math import factorial
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from queuingSystemModel import QueuingSystemModel

CHANNELS_NUMBER = 2
PROCESSING_FLOW_RATE = 4
APPLICATIONS_FLOW_RATE = 3
WAITING_FLOW_RATE = 1
MAX_QUEUE_LENGTH = 2
SIMULATIONS_COUNT = 100


def calculate_empiric_probabilities(model, applications_flow_rate):
    applications_done = np.array(model.processed_applications)
    rejected_applications = np.array(model.rejected_applications)
    np.array(model.applications_in_queue)

    total_applications_amount = len(applications_done) + len(rejected_applications)

    probabilities_of_rejection = len(rejected_applications) / total_applications_amount
    probability_of_finished = 1 - probabilities_of_rejection
    flow_rate_value = applications_flow_rate * probability_of_finished

    print('Empiric Probabilities')
    print(pd.DataFrame({
        'Rejection': [probabilities_of_rejection],
        'Finished': [probability_of_finished],
        'Flow Rate': [flow_rate_value],
    }))

    print('\nAverage values')
    print(pd.DataFrame({
        'queue length': [np.array(model.applications_in_queue).mean()],
        'qs length': [np.array(model.total_applications).mean()],
        'queue time': [np.array(model.applications_in_queue_time).mean()],
        'busy channels': [probability_of_finished * APPLICATIONS_FLOW_RATE / PROCESSING_FLOW_RATE],
        'wait time': [np.array(model.applications_QS_times).mean()],
    }))


def calculate_average_values(ro, betta, pn, initial_probability, channels_count, max_queue_length):
    get_channels_product = lambda index: np.prod([channels_count + t * betta for t in range(1, index + 1)])

    average_queue_length = sum([index * pn * (ro ** index) / get_channels_product(index) for index in range(1, max_queue_length + 1)])

    value_of_channels = sum([index * initial_probability * (ro ** index) / factorial(index) for index in range(1, channels_count + 1)])

    average_applications_in_QS = value_of_channels + sum(
        [(channels_count + index) * pn * ro ** index / get_channels_product(index) for index in range(1, max_queue_length + 1)])

    return average_queue_length, average_applications_in_QS


def calculate_theoretical_probabilities(
        channels_count,
        max_queue_length,
        applications_flow_rate,
        processing_flow_rate,
        queue_waiting_flow_rate,
):
    ro = applications_flow_rate / processing_flow_rate
    betta = queue_waiting_flow_rate / processing_flow_rate

    get_channels_product = lambda index: np.prod([channels_count + t * betta for t in range(1, index + 1)])

    value_of_channels = sum([ro ** i / factorial(i) for i in range(channels_count + 1)])
    value_of_queue = sum([ro ** index / get_channels_product(index) for index in range(1, max_queue_length + 1)])
    initial_probability = (value_of_channels + (ro ** channels_count / factorial(channels_count)) * value_of_queue) ** -1

    px = (ro ** channels_count / factorial(channels_count)) * initial_probability
    pn = px

    probabilities_of_rejection = (ro ** max_queue_length / get_channels_product(max_queue_length)) * pn
    probability_of_finished = 1 - probabilities_of_rejection
    flow_rate_value = probability_of_finished * applications_flow_rate

    average_queue_length, average_applications_in_QS = calculate_average_values(
        ro,
        betta,
        pn,
        initial_probability,
        channels_count,
        max_queue_length,
    )

    print('\n\nTheoretical Probabilities')
    print(pd.DataFrame({
        'Rejection': [probabilities_of_rejection],
        'Finished': [probability_of_finished],
        'Flow Rate': [flow_rate_value],
    }))

    print('\nAverage values')
    print(pd.DataFrame({
        'queue length': [average_queue_length],
        'qs length': [average_applications_in_QS],
        'queue time': [probability_of_finished * ro / applications_flow_rate],
        'busy channels': [probability_of_finished * ro],
        'wait time': [average_applications_in_QS / applications_flow_rate],
    }))


def show_plot(data, title):
    fig, axs = plt.subplots(1)
    axs.hist(np.array(data), 100)
    axs.set_title(title)
    plt.show()


if __name__ == '__main__':
    model = QueuingSystemModel(CHANNELS_NUMBER, PROCESSING_FLOW_RATE, WAITING_FLOW_RATE)
    model.run(10_000, MAX_QUEUE_LENGTH, APPLICATIONS_FLOW_RATE)

    calculate_empiric_probabilities(model, APPLICATIONS_FLOW_RATE)
    calculate_theoretical_probabilities(
        CHANNELS_NUMBER,
        MAX_QUEUE_LENGTH,
        APPLICATIONS_FLOW_RATE,
        PROCESSING_FLOW_RATE,
        WAITING_FLOW_RATE,
    )

    show_plot(model.applications_QS_times, 'Wait times')
    show_plot(model.total_applications, 'total applications')
