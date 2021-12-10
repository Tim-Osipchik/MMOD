from math import factorial
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
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

    value_of_channels = sum([ro ** index / factorial(index) for index in range(channels_count + 1)])
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
    print('P0:', initial_probability)
    probabilities = [initial_probability]
    for index in range(1, channels_count + 1):
        probabilities.append((ro ** index / factorial(index)) * initial_probability)
        print('P' + str(index) + ': ' + str(probabilities[-1]))

    last_channel_prob = probabilities[-1]
    for index in range(1, max_queue_length):
        probabilities.append((ro ** index / get_channels_product(index)) * last_channel_prob)
        print('P' + str(channels_count + index) + ': ' + str(probabilities[-1]))

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

    return probabilities


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
    theoretical_probabilities = calculate_theoretical_probabilities(
        channels_count=channels_number,
        max_queue_length=max_queue_length,
        applications_flow_rate=applications_flow_rate,
        processing_flow_rate=processing_flow_rate,
        queue_waiting_flow_rate=waiting_flow_rate,
    )

    chi2, p, dof, ex = chi2_contingency(np.array([theoretical_probabilities, empiric_probabilities]))
    print('chi2: ', p)

    show_plot(model.applications_QS_times, 'Wait times')
    show_final_probabilities(
        empiric_probabilities,
        theoretical_probabilities,
        'processing_flow_rate = ' + str(processing_flow_rate),
    )


if __name__ == '__main__':
    run_simulation(
        channels_number=1,
        max_queue_length=4,
        processing_flow_rate=1,
        waiting_flow_rate=2,
        applications_flow_rate=6,
        simulations_count=10_000,
    )

    # run_simulation(
    #     channels_number=1,
    #     max_queue_length=4,
    #     processing_flow_rate=2,
    #     waiting_flow_rate=2,
    #     applications_flow_rate=4,
    #     simulations_count=10_000,
    # )
    #
    # run_simulation(
    #     channels_number=1,
    #     max_queue_length=4,
    #     processing_flow_rate=4,
    #     waiting_flow_rate=2,
    #     applications_flow_rate=4,
    #     simulations_count=10_000,
    # )

    # run_simulation(
    #     channels_number=2,
    #     max_queue_length=3,
    #     processing_flow_rate=2,
    #     waiting_flow_rate=3,
    #     applications_flow_rate=3,
    #     simulations_count=10_000,
    # )
    #
    # run_simulation(
    #     channels_number=2,
    #     max_queue_length=3,
    #     processing_flow_rate=2,
    #     waiting_flow_rate=3,
    #     applications_flow_rate=5,
    #     simulations_count=10_000,
    # )
    #
    # run_simulation(
    #     channels_number=2,
    #     max_queue_length=3,
    #     processing_flow_rate=2,
    #     waiting_flow_rate=3,
    #     applications_flow_rate=7,
    #     simulations_count=10_000,
    # )
