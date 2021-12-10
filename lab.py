import math
from math import factorial
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from queuingSystem import QueuingSystem


def calculate_empiric_probabilities(model, applications_flow_rate, processing_flow_rate, max_queue_length):
    applications_done = np.array(model.processed_applications)
    rejected_applications = np.array(model.rejected_applications)
    np.array(model.applications_in_queue)

    total_applications_amount = len(applications_done) + len(rejected_applications)

    probabilities_of_rejection = len(rejected_applications) / total_applications_amount
    probability_of_finished = 1 - probabilities_of_rejection
    flow_rate_value = applications_flow_rate * probability_of_finished

    print('Empiric Probabilities')

    probabilities = []
    count = 0
    for value in range(1, model.channels_number + max_queue_length + 1):
        probabilities.append(len(applications_done[applications_done == value]) / total_applications_amount)
        print('P' + str(count) + ': ' + str(probabilities[-1]))
        count = count + 1

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
        'busy channels': [probability_of_finished * applications_flow_rate / processing_flow_rate],
        'wait time': [np.array(model.applications_QS_times).mean()],
    }))

    return probabilities


def calculate_average_values(ro, betta, pn, initial_probability, channels_count, max_queue_length):
    get_channels_product = lambda index: np.prod([channels_count + t * betta for t in range(1, index + 1)])

    average_queue_length = sum([index * pn * (ro ** index) / get_channels_product(index) for index in range(1, max_queue_length + 1)])

    value_of_channels = sum([index * initial_probability * (ro ** index) / factorial(index) for index in range(1, channels_count + 1)])

    average_applications_in_QS = value_of_channels + sum(
        [(channels_count + index) * pn * ro ** index / get_channels_product(index) for index in range(1, max_queue_length + 1)])

    return average_queue_length, average_applications_in_QS


def calculate_theoretical_probabilities(applications_flow_rate, processing_flow_rate):
    ro = applications_flow_rate / processing_flow_rate
    coefficient_of_variation = 1 / np.sqrt([3])

    average_queue_length = ro ** 2 * (1 + coefficient_of_variation ** 2) / (2 * (1 - ro))
    average_applications_in_QS = average_queue_length + ro
    average_waiting_time = average_queue_length / applications_flow_rate
    average_time_in_QS = average_applications_in_QS / applications_flow_rate

    p0 = 1 - ro
    p1 = ro * p0
    p_rejection = 1 - p1

    print('\n\nTheoretical Probabilities')
    print('\nAverage values')
    print(pd.DataFrame({
        'queue length': np.round(average_queue_length, 2),
        'qs length': np.round(average_applications_in_QS, 2),
        'time in qs': np.round(average_time_in_QS, 2),
        'wait time': np.round(average_waiting_time, 2),
    }))

def show_plot(data, title):
    fig, axs = plt.subplots()
    axs.hist(np.array(data), 100)
    axs.set_title(title)
    plt.show()


def show_final_probabilities(empiric_probabilities, theoretical_probabilities, title):
    fig, axs = plt.subplots()
    axs.bar([i - 0.2 for i in range(len(empiric_probabilities))], empiric_probabilities, width=0.4, label='empiric')
    axs.bar([i + 0.2 for i in range(len(theoretical_probabilities))], theoretical_probabilities, width=0.4, label='theoretical')
    plt.title('Final probabilities. ' + title)
    plt.legend()
    plt.show()


def run_simulation(
    channels_number,
    max_queue_length,
    processing_flow_rate,
    waiting_flow_rate,
    applications_flow_rate,
    simulations_count,
):
    print('\n\nchannels_number: ', channels_number)
    print('processing_flow_rate: ', processing_flow_rate)
    print('applications_flow_rate: ', applications_flow_rate)
    print('max_queue_length: ', max_queue_length)
    print('waiting_flow_rate: ', waiting_flow_rate)

    model = QueuingSystem(
        channels_number=channels_number,
        processing_flow_rate=processing_flow_rate,
        queue_waiting_flow_rate=waiting_flow_rate,
        applications_flow_rate=applications_flow_rate,
    )
    model.run(simulations_count)

    empiric_probabilities = calculate_empiric_probabilities(
        model,
        applications_flow_rate,
        processing_flow_rate,
        max_queue_length,
    )
    calculate_theoretical_probabilities(
        applications_flow_rate=applications_flow_rate,
        processing_flow_rate=processing_flow_rate,
    )

    show_plot(model.applications_QS_times, 'Wait times')


if __name__ == '__main__':
    run_simulation(
        channels_number=1,
        max_queue_length=1,
        processing_flow_rate=0.125,
        waiting_flow_rate=None,
        applications_flow_rate=0.1,
        simulations_count=10_000,
    )
